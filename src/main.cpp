/******************************************************************************
 *                                                                            *
 * Copyright (C) 2018 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file main.cpp
 * @authors: Fabrizio Bottarel <fabrizio.bottarel@iit.it>
 */

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/math/Math.h>

#include <vtkSmartPointer.h>
#include <vtkCommand.h>
#include <vtkProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkSuperquadric.h>
#include <vtkTransform.h>
#include <vtkSampleFunction.h>
#include <vtkContourFilter.h>
#include <vtkActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAxesActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleSwitch.h>
#include <vtkMatrix4x4.h>

#include <objectsLib.h>

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;

Mutex mutex;

/****************************************************************/

class UpdateCommand : public vtkCommand
{
    const bool *closing;

public:
    /****************************************************************/
    vtkTypeMacro(UpdateCommand, vtkCommand);

    /****************************************************************/
    static UpdateCommand *New()
    {
        return new UpdateCommand;
    }

    /****************************************************************/
    UpdateCommand() : closing(nullptr) { }

    /****************************************************************/
    void set_closing(const bool &closing)
    {
        this->closing=&closing;
    }

    /****************************************************************/
    void Execute(vtkObject *caller, unsigned long vtkNotUsed(eventId),
                 void *vtkNotUsed(callData))
    {
        LockGuard lg(mutex);
        vtkRenderWindowInteractor* iren=static_cast<vtkRenderWindowInteractor*>(caller);
        if (closing!=nullptr)
        {
            if (*closing)
            {
                iren->GetRenderWindow()->Finalize();
                iren->TerminateApp();
                return;
            }
        }

        //iren->GetRenderWindow()->SetWindowName("Grasping pose candidates");
        iren->Render();
    }
};

/****************************************************************/

class GraspProcessorModule : public RFModule
{

    class GraspPose
    {
    public:
        //  essential parameters for representing a grasping pose
        vtkSmartPointer<vtkAxesActor> pose_vtk_actor;
        vtkSmartPointer<vtkTransform> pose_vtk_transform;
        Matrix pose_transform;
        Matrix pose_rotation;
        Vector pose_translation;
        Vector pose_ax_size;

        GraspPose() : pose_transform(4,4), pose_rotation(3,3), pose_translation(3), pose_ax_size(3)
        {
            pose_transform.eye();
            pose_rotation.eye();
            pose_translation.zero();
            pose_ax_size.zero();
            pose_vtk_actor = vtkSmartPointer<vtkAxesActor>::New();
            pose_vtk_transform = vtkSmartPointer<vtkTransform>::New();
        }

        //  methods to be defined

    };



    string moduleName;

    RpcClient superq_rpc;
    RpcClient point_cloud_rpc;
    RpcClient action_render_rpc;
    RpcServer module_rpc;   //will be replaced by idl services

    bool closing;

    //  visualization objects
    unique_ptr<Points> vtk_points;
    unique_ptr<Superquadric> vtk_superquadric;

    vtkSmartPointer<vtkRenderer> vtk_renderer;
    vtkSmartPointer<vtkRenderWindow> vtk_renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> vtk_renderWindowInteractor;
    vtkSmartPointer<vtkAxesActor> vtk_axes;
    vtkSmartPointer<vtkOrientationMarkerWidget> vtk_widget;
    vtkSmartPointer<vtkCamera> vtk_camera;
    vtkSmartPointer<vtkInteractorStyleSwitch> vtk_style;
    vtkSmartPointer<UpdateCommand> vtk_updateCallback;

    //  grasping pose candidates
    vector<GraspPose> pose_candidates;

    //  filtering constants
    double table_height_z;
    double palm_width_y;
    double grasp_width_x;

    bool configure(ResourceFinder &rf) override
    {

        //  set module name
        if (rf.check("name"))
        {
            moduleName = rf.find("name").asString();
        }
        else
        {
            moduleName = "graspProcessor";
        }

        //  open the necessary ports
        superq_rpc.open("/" + moduleName + "/superquadricRetrieve:rpc");
        point_cloud_rpc.open("/" + moduleName + "/pointCloud:rpc");
        action_render_rpc.open("/" + moduleName + "/actionRenderer:rpc");
        module_rpc.open("/" + moduleName + "/cmd:rpc");

        //  attach callback
        attach(module_rpc);

        //  initialize an empty point cloud to display
        PointCloud<DataXYZRGBA> pc;
        pc.clear();
        vtk_points = unique_ptr<Points>(new Points(pc, 3));

        //  initialize a zero-superquadric to display (cyan coloured)
        Vector r(11, 0.0);
        vtk_superquadric = unique_ptr<Superquadric>(new Superquadric(r, 1.2));

        //  set up rendering window and interactor
        vtk_renderer = vtkSmartPointer<vtkRenderer>::New();
        vtk_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        vtk_renderWindow->SetSize(600,600);
        vtk_renderWindow->AddRenderer(vtk_renderer);
        vtk_renderWindowInteractor=vtkSmartPointer<vtkRenderWindowInteractor>::New();
        vtk_renderWindowInteractor->SetRenderWindow(vtk_renderWindow);

        //  set up point cloud and superquadric actors
        vtk_renderer->AddActor(vtk_points->get_actor());
        vtk_renderer->AddActor(vtk_superquadric->get_actor());

        //  set a neutral color for the background
        vtk_renderer->SetBackground(0.1, 0.2, 0.2);

        //  set up root reference frame axes widget
        vtk_axes = vtkSmartPointer<vtkAxesActor>::New();
        vtk_widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        vtk_widget->SetOutlineColor(0.9300,0.5700,0.1300);
        vtk_widget->SetOrientationMarker(vtk_axes);
        vtk_widget->SetInteractor(vtk_renderWindowInteractor);
        vtk_widget->SetViewport(0.0,0.0,0.2,0.2);
        vtk_widget->SetEnabled(1);
        vtk_widget->InteractiveOn();

        //  set up the camera position according to the point cloud (initially empty)
        vector<double> pc_bounds(6), pc_centroid(3);
        vtk_points->get_polydata()->GetBounds(pc_bounds.data());

        for (size_t i=0; i<pc_centroid.size(); i++)
        {
            pc_centroid[i] = 0.5*(pc_bounds[i<<1]+pc_bounds[(i<<1)+1]);
        }

        vtk_camera = vtkSmartPointer<vtkCamera>::New();
        vtk_camera->SetPosition(pc_centroid[0]+1.0, pc_centroid[1], pc_centroid[2]+0.5);
        vtk_camera->SetViewUp(0.0, 0.0, 1.0);
        vtk_renderer->SetActiveCamera(vtk_camera);

        //  activate interactor
        vtk_style=vtkSmartPointer<vtkInteractorStyleSwitch>::New();
        vtk_style->SetCurrentStyleToTrackballCamera();
        vtk_renderWindowInteractor->SetInteractorStyle(vtk_style);

        vtk_renderWindowInteractor->Initialize();
        vtk_renderWindowInteractor->CreateRepeatingTimer(10);

        //  set up the visualizer refresh callback
        vtk_updateCallback = vtkSmartPointer<UpdateCommand>::New();
        vtk_updateCallback->set_closing(closing);
        vtk_renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, vtk_updateCallback);

        //  start the interactor and renderer
        vtk_renderWindowInteractor->GetRenderWindow()->SetWindowName("Grasping pose candidates");
        vtk_renderWindowInteractor->Render();
        vtk_renderWindowInteractor->Start();

        return true;

    }

    /****************************************************************/
    bool updateModule() override
    {
        return false;
    }

    /****************************************************************/
    double getPeriod() override
    {
        return 1.0;
    }

    /****************************************************************/
    bool interruptModule() override
    {
        superq_rpc.interrupt();
        point_cloud_rpc.interrupt();
        action_render_rpc.interrupt();
        module_rpc.interrupt();
        closing = true;

        return true;

    }

    /****************************************************************/
    bool close() override
    {
        superq_rpc.close();
        point_cloud_rpc.close();
        action_render_rpc.close();
        module_rpc.close();

        return true;

    }

    /****************************************************************/
    bool respond(const Bottle& command, Bottle& reply) override
    {
        string obj = command.get(0).asString();
        yDebug() << "Requested object: " << obj;
        PointCloud<DataXYZRGBA> pc;
        pc.clear();
        if (requestRefreshPointCloud(pc, obj))
        {
            if(requestRefreshSuperquadric(pc))
            {
                reply.addString("ack");
                return true;
            }
        }

        reply.addString("nack");

        return false;

    }

    /****************************************************************/
    void refreshPointCloud(const PointCloud<DataXYZRGBA> &points)
    {
       if (points.size() > 0)
       {
           LockGuard lg(mutex);

           //   set the vtk point cloud object with the read data
           vtk_points->set_points(points);
           vtk_points->set_colors(points);

       }
    }

    /****************************************************************/
    void refreshSuperquadric(const Vector superq_params)
    {
        //  the incoming message has the following syntax
        //  (center-x center-y center-z angle size-x size-y size-z epsilon-1 epsilon-2)
        //  we need to reformat the vector in the format
        //  (dimensions (x0 x1 x2)) (exponents (x3 x4)) (center (x5 x6 x7)) (orientation (x8 x9 x10 x11))

        Vector superq_params_sorted;

        superq_params_sorted(0) = superq_params(4);
        superq_params_sorted(1) = superq_params(5);
        superq_params_sorted(2) = superq_params(6);
        superq_params_sorted(3) = superq_params(7);
        superq_params_sorted(4) = superq_params(8);
        superq_params_sorted(5) = superq_params(0);
        superq_params_sorted(6) = superq_params(1);
        superq_params_sorted(7) = superq_params(2);
        superq_params_sorted(8) = superq_params(3);
        superq_params_sorted(9) = 0.0;
        superq_params_sorted(10) = 0.0;
        superq_params_sorted(11) = 1.0;

        LockGuard lg(mutex);

        vtk_superquadric ->set_parameters(superq_params_sorted);

    }

    /****************************************************************/
    bool requestRefreshPointCloud(PointCloud<DataXYZRGBA> point_cloud, const string &object)
    {
        //  query point-cloud-read via rpc for the point cloud
        //  command: get_point_cloud objectName
        //  put point cloud into container, return true if operation was ok
        //  or call refreshpointcloud
        Bottle cmd_request;
        cmd_request.clear();

        cmd_request.addString("get_point_cloud");
        cmd_request.addString(object);

        point_cloud_rpc.write(cmd_request, point_cloud);

        if (point_cloud.size() > 0)
        {
            yDebug() << "Point cloud retrieved; contains " << point_cloud.size() << "points";
            refreshPointCloud(point_cloud);
            return true;
        }
        else
        {
            yError() << "Point cloud null or empty";
            return false;
        }

    }

    /****************************************************************/
    bool requestRefreshSuperquadric(PointCloud<DataXYZRGBA> &point_cloud)
    {
        //  query find-superquadric via rpc for the superquadric
        //  command: (point cloud)
        //  parse the reply (center-x center-y center-z angle size-x size-y size-z epsilon-1 epsilon-2)
        //  refresh superquadric with parameters
        Bottle sq_reply;
        sq_reply.clear();

        superq_rpc.write(point_cloud, sq_reply);

        Vector superq_parameters;
        sq_reply.write(superq_parameters);

        if (superq_parameters.size() == 9)
        {
            refreshSuperquadric(superq_parameters);
            return true;
        }
        else
        {
            yError() << "Retrieved superquadric is invalid! " << superq_parameters.toString();
            return false;
        }

    }

    /****************************************************************/
    void computeGraspCandidates()
    {


    }
































};

int main(int argc, char *argv[])
{
    Network yarp;
    ResourceFinder rf;
    rf.configure(argc, argv);

    if (!yarp.checkNetwork())
    {
        yError() << "YARP network not detected. Check nameserver";
        return EXIT_FAILURE;
    }

    GraspProcessorModule disp;

    return disp.runModule(rf);
}
