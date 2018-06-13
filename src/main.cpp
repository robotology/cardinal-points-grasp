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

        iren->GetRenderWindow()->SetWindowName("Grasping pose candidates");
        iren->Render();
    }
};

/****************************************************************/

class DisplaySuperQ : public RFModule, RateThread
{
    enum class SuperquadricType
    {
        //  we need to handle different superquadrics at once
        ANALYTICAL_GRAD_SQ,
        FINITE_DIFF_SQ
    };

    struct grasp_axis {

        vtkSmartPointer<vtkAxesActor> ax_actor;
        vtkSmartPointer<vtkTransform> ax_transform;
        Matrix ax_orientation;
        Vector ax_size;

        grasp_axis(): ax_orientation(3,3), ax_size(3) {
            ax_actor = vtkSmartPointer<vtkAxesActor>::New();
            ax_transform = vtkSmartPointer<vtkTransform>::New();
        }
    };

    class PointCloudProcessor: public TypedReaderCallback<PointCloud<DataXYZRGBA>>
    {
        DisplaySuperQ *displayer;
        using TypedReaderCallback<PointCloud<DataXYZRGBA>>::onRead;
        virtual void onRead(PointCloud<DataXYZRGBA> &pointCloud)
        {
            //  define what to do when the callback is triggered
            //  i.e. when a point cloud comes in
            displayer->refreshPointCloud(pointCloud);
            //  TODO: a point cloud that gets read here should trigger the acquisition of
            //  both superquadrics with rpcs. We don't have the rpc from superquadric-model
            //  fixed yet so we only trigger the acquisition from find-superquadric
            //  via displayer->refreshSuperquadric
            //  prepare cmd, reply for find-superquadric
            //  write on rpc port
            //  convert syntax of response into a Property
            //  call refreshSuperquadric
            Bottle sq_reply;
            sq_reply.clear();

            displayer->superq2RPC.write(pointCloud, sq_reply);
            Property superquadric_params;

            Bottle bottle;
            Bottle &b1=bottle.addList();
            b1.addDouble(sq_reply.get(4).asDouble());
            b1.addDouble(sq_reply.get(5).asDouble());
            b1.addDouble(sq_reply.get(6).asDouble());
            superquadric_params.put("dimensions", bottle.get(0));

            Bottle &b2=bottle.addList();
            b2.addDouble(sq_reply.get(7).asDouble());
            b2.addDouble(sq_reply.get(8).asDouble());
            superquadric_params.put("exponents", bottle.get(1));

            Bottle &b3=bottle.addList();
            b3.addDouble(sq_reply.get(0).asDouble());
            b3.addDouble(sq_reply.get(1).asDouble());
            b3.addDouble(sq_reply.get(2).asDouble());
            superquadric_params.put("center", bottle.get(2));

            Bottle &b4=bottle.addList();
            b4.addDouble(sq_reply.get(3).asDouble()); b4.addDouble(0.0); b4.addDouble(0.0); b4.addDouble(1.0);
            superquadric_params.put("orientation", bottle.get(3));

            displayer->refreshSuperquadric(superquadric_params, SuperquadricType::ANALYTICAL_GRAD_SQ);

        }
    public:
        PointCloudProcessor(DisplaySuperQ *displayer_) : displayer(displayer_) { }
    };

    class SuperquadricProcessor: public TypedReaderCallback<Property>
    {
        DisplaySuperQ *displayer;
        SuperquadricType sq_type;
        using TypedReaderCallback<Property>::onRead;
        virtual void onRead(Property &superQ)
        {
            //  define what to do when the callback is triggered
            //  i.e. when a superquadric comes in
            displayer->refreshSuperquadric(superQ, sq_type);
        }
    public:
        SuperquadricProcessor(DisplaySuperQ *displayer_, SuperquadricType type) : displayer(displayer_), sq_type(type) { }
    };

    PointCloudProcessor PCproc;
    SuperquadricProcessor SQprocFD, SQprocAG;   //  FD = finite differences, AG = analytical gradient

    BufferedPort<PointCloud<DataXYZRGBA>> pointCloudInPort;
    BufferedPort<Property> superq1InPort;
    BufferedPort<Property> superq2InPort;

    RpcClient superq1RPC;        //  not needed for now!
    RpcClient superq2RPC;

    string moduleName;

    bool closing;

    unique_ptr<Points> vtk_points;
    unique_ptr<Superquadric> vtk_superquadric_fin_diff;
    unique_ptr<Superquadric> vtk_superquadric_an_grad;

    vtkSmartPointer<vtkRenderer> vtk_renderer;
    vtkSmartPointer<vtkRenderWindow> vtk_renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> vtk_renderWindowInteractor;
    vtkSmartPointer<vtkAxesActor> vtk_axes;
    vtkSmartPointer<vtkOrientationMarkerWidget> vtk_widget;
    vtkSmartPointer<vtkCamera> vtk_camera;
    vtkSmartPointer<vtkInteractorStyleSwitch> vtk_style;
    vtkSmartPointer<UpdateCommand> vtk_updateCallback;

    std::vector<grasp_axis> pose_candidates;
    const double TABLE_HEIGHT_Z;

    bool configure(ResourceFinder &rf) override
    {

        if (rf.check("name"))
        {
            moduleName = rf.find("name").asString();
        }
        else
        {
            moduleName = "superq-cloud-display";
        }

        //  attach callbacks to ports
        pointCloudInPort.useCallback(PCproc);
        superq1InPort.useCallback(SQprocFD);
        superq2InPort.useCallback(SQprocAG);

        //  open ports
        pointCloudInPort.open("/" + moduleName + "/pointCloud:i");
        superq1InPort.open("/" + moduleName + "/superquadricFiniteDiff:i");
        superq2InPort.open("/" + moduleName + "/superquadricAnalyticalGrad:i");
        //superq1RPC.open("/" + moduleName + "/superquadricFiniteDiff:rpc");
        superq2RPC.open("/" + moduleName + "/superquadricAnalyticalGrad:rpc");

        //  initialize empty point cloud to display
        PointCloud<DataXYZRGBA> pc;
        pc.clear();
        vtk_points = unique_ptr<Points>(new Points(pc, 3));

        //  initialize superquadric
        Vector r(11, 0.0);
        vtk_superquadric_fin_diff = unique_ptr<Superquadric>(new Superquadric(r, 2.0));
        vtk_superquadric_an_grad = unique_ptr<Superquadric>(new Superquadric(r, 1.2));

        //  set up renderer window
        vtk_renderer = vtkSmartPointer<vtkRenderer>::New();
        vtk_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        vtk_renderWindow->SetSize(600,600);
        vtk_renderWindow->AddRenderer(vtk_renderer);
        vtk_renderWindowInteractor=vtkSmartPointer<vtkRenderWindowInteractor>::New();
        vtk_renderWindowInteractor->SetRenderWindow(vtk_renderWindow);

        //  set up actors for the superquadric and point cloud
        vtk_renderer->AddActor(vtk_points->get_actor());
        vtk_renderer->AddActor(vtk_superquadric_fin_diff->get_actor());
        vtk_renderer->AddActor(vtk_superquadric_an_grad->get_actor());
        vtk_renderer->SetBackground(0.1,0.2,0.2);

        //  set up axes widget
        vtk_axes = vtkSmartPointer<vtkAxesActor>::New();
        vtk_widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        vtk_widget->SetOutlineColor(0.9300,0.5700,0.1300);
        vtk_widget->SetOrientationMarker(vtk_axes);
        vtk_widget->SetInteractor(vtk_renderWindowInteractor);
        vtk_widget->SetViewport(0.0,0.0,0.2,0.2);
        vtk_widget->SetEnabled(1);
        vtk_widget->InteractiveOn();

        //  set up the camera position according to the point cloud distribution
        vector<double> pc_bounds(6), pc_centroid(3);
        vtk_points->get_polydata()->GetBounds(pc_bounds.data());
        for (size_t i=0; i<pc_centroid.size(); i++)
            pc_centroid[i] = 0.5*(pc_bounds[i<<1]+pc_bounds[(i<<1)+1]);

        vtk_camera = vtkSmartPointer<vtkCamera>::New();
        vtk_camera->SetPosition(pc_centroid[0]+1.0, pc_centroid[1], pc_centroid[2]+0.5);
        vtk_camera->SetViewUp(0.0, 0.0, 1.0);
        vtk_renderer->SetActiveCamera(vtk_camera);

        vtk_style=vtkSmartPointer<vtkInteractorStyleSwitch>::New();
        vtk_style->SetCurrentStyleToTrackballCamera();
        vtk_renderWindowInteractor->SetInteractorStyle(vtk_style);

        vtk_renderWindowInteractor->Initialize();
        vtk_renderWindowInteractor->CreateRepeatingTimer(10);

        //  set up the visualizer refresh callback
        vtk_updateCallback = vtkSmartPointer<UpdateCommand>::New();
        vtk_updateCallback->set_closing(closing);
        vtk_renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, vtk_updateCallback);

        this->start();

        vtk_renderWindowInteractor->Start();

        return true;

    }

    /****************************************************************/
    bool updateModule() override
    {
        //  do something during update?
        return true;

    }

    /****************************************************************/
    bool interruptModule() override
    {
        superq1InPort.interrupt();
        superq2InPort.interrupt();
        //superq1RPC.interrupt();
        superq2RPC.interrupt();
        pointCloudInPort.interrupt();
        closing = true;

        //  interrupt the RateThread loop
        this->stop();

        return true;

    }

    /****************************************************************/
    bool close() override
    {
        if (!superq1InPort.isClosed())
            superq1InPort.close();
//        if (!superq1RPC.isClosed())
//            superq1RPC.close();
        superq2RPC.close();
        if (!superq2InPort.isClosed())
            superq2InPort.close();
        if (!pointCloudInPort.isClosed())
            pointCloudInPort.close();

        return true;

    }

    /****************************************************************/
    virtual bool threadInit() override
    {
        yInfo() << "Grasp computation thread starting...";
        return true;
    }

    /****************************************************************/
    virtual void afterStart(bool s) override
    {
        if (s)
            yInfo() << "Grasp computation thread started.";
        else
            yInfo() << "Grasp computation thread failed to start!";
    }

    /****************************************************************/
    virtual void run() override
    {
        refreshGraspComputation();
    }

    /****************************************************************/
    virtual void threadRelease() override
    {
        yInfo() << "Grasp computation thread terminating.";
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
    void refreshSuperquadric(const Property &superq, const SuperquadricType sq_type)
    {
        //  the incoming message has the following syntax
        //  (dimensions (x0 x1 x2)) (exponents (x3 x4)) (center (x5 x6 x7)) (orientation (x8 x9 x10 x11))

        //  Show superquadric info for debugging
        yInfo() << "Superquadric: " << superq.toString();

        //  pass the parameters as a single vector
        Vector params;

        Bottle *superq_dimension = superq.find("dimensions").asList();
        if (!superq_dimension->isNull())
            for (int i=0; i<superq_dimension->size(); i++)
                params.push_back(superq_dimension->get(i).asDouble());

        Bottle *superq_exponents = superq.find("exponents").asList();
        if (!superq_exponents->isNull())
            for (int i=0; i<superq_exponents->size(); i++)
                params.push_back(superq_exponents->get(i).asDouble());

        Bottle *superq_center = superq.find("center").asList();
        if (!superq_center->isNull())
            for (int i=0; i<superq_center->size(); i++)
                params.push_back(superq_center->get(i).asDouble());

        Bottle *superq_orientation = superq.find("orientation").asList();
        if (!superq_orientation->isNull())
            for (int i=0; i<superq_orientation->size(); i++)
                params.push_back(superq_orientation->get(i).asDouble());

        //  check whether we have a sufficient number of parameters
        if (params.size() == 12)
            switch (sq_type)
            {
            case SuperquadricType::FINITE_DIFF_SQ:
                vtk_superquadric_fin_diff->set_parameters(params);
                break;
            case SuperquadricType::ANALYTICAL_GRAD_SQ:
                vtk_superquadric_an_grad->set_parameters(params);
                break;
            }
        else
            yError() << "Invalid superquadric";

        refreshGraspComputation();

    }

    /****************************************************************/
    void refreshGraspComputation()
    {
        //  extract parameters from superquadric, compute possible grasps with simple heuristics
        LockGuard lg(mutex);

        for (grasp_axis candidate: pose_candidates)
        {
            vtk_renderer->RemoveActor(candidate.ax_actor);
        }

        pose_candidates.clear();

        //  get center, orientation and axes size
        Vector superquadric_center = vtk_superquadric_an_grad->getCenter();
        Vector superquadric_WXYZorientation = vtk_superquadric_an_grad->getOrientationWXYZ();
        Vector superquadric_axis_size = vtk_superquadric_an_grad->getAxesSize();

        //  get rotation matrix wrt root reference frame
        superquadric_WXYZorientation(3) = superquadric_WXYZorientation(3) / (180/M_PI);
        Matrix superquadric_orientation = yarp::math::axis2dcm(superquadric_WXYZorientation).submatrix(0, 2, 0, 2);
        yDebug() << "Superquadric orientation: " << superquadric_orientation.toString();

        //  extract axes orientations and size wrt root ref frame
        Vector sq_axis_x = superquadric_orientation.getCol(0);
        Vector sq_axis_y = superquadric_orientation.getCol(1);
        Vector sq_axis_z = superquadric_orientation.getCol(2);

        using namespace yarp::math;

        //  create suitable candidates for score evaluation
        //  first create the search space for gx
        //  then create search space for gy
        std::vector<Vector> search_space_gx = {sq_axis_x, -1*sq_axis_x, sq_axis_y, -1*sq_axis_y};
        std::vector<Vector> search_space_gy = {sq_axis_x, -1*sq_axis_x, sq_axis_y, -1*sq_axis_y, sq_axis_z, -1*sq_axis_z};

        //  create pose candidates
        for (size_t idx = 0; idx < search_space_gx.size(); idx++)
        {
            Vector gx = search_space_gx[idx];
            //  TODO: filter candidates wrt grasp width
            //  for each candidate gx, generate a number of candidates for gy
            for (size_t jdx = 0; jdx < search_space_gy.size(); jdx++)
            {
                Vector gy = search_space_gy[jdx];
                //  keep candidate if orthogonal to gx
                //  TODO: filter candidate wrt palm width
                if (dot(gx,gy)*dot(gx,gy) < 0.0001)
                {
                    //  if candidate is good, set gx, gy
                    grasp_axis candidate;
                    candidate.ax_orientation.setCol(0, gx);
                    candidate.ax_orientation.setCol(1, gy);
                    //  set gz
                    yDebug() << "Candidate gx: " << gx.toString() << " Candidate gy: " << gy.toString();
                    yDebug() << "Cross product: " << cross(gx,gy).toString();
                    candidate.ax_orientation.setCol(2, cross(gx, gy));
                    //  set axes size
                    candidate.ax_size(0) = superquadric_axis_size(idx/2);   // superquadric size along gx
                    candidate.ax_size(1) = superquadric_axis_size(jdx/2);   // superquadric size along gy
                    candidate.ax_size(2) = superquadric_axis_size(3 - idx/2 - jdx/2);   //THIS IS VERY DODGY; THERE MIGHT BE A BUG HERE
                    pose_candidates.push_back(candidate);
                }
            }
        }

        //  add all candidates to render window
        //  TODO: first generate all candidates, then filter them according to their gx and gy size, and position wrt table
        std::vector<grasp_axis> filtered_candidates;

        for (grasp_axis candidate:pose_candidates)
        {
            //  make a 4x4 zero-filled matrix and put in the 3x3 rotation of gx, gy, gz
            Matrix transform_matrix(4,4);
            transform_matrix.zero();
            transform_matrix.setSubmatrix(candidate.ax_orientation, 0, 0);

            //  make the 4x4 homogeneous
            transform_matrix(3,3) = 1;

            //  the translation part of the 4x4 corresponds to the superquadric center
            yDebug() << "Superquadric center: " << superquadric_center.toString();
            transform_matrix.setSubcol(superquadric_center, 0, 3);

            //  shift along gz by the superquadric size along such axis
            //  assuming right hand grasp: move the grasp along -gx
            Vector gz = candidate.ax_orientation.getCol(2);
            double z_offset = candidate.ax_size(2);
            yDebug() << "Homogeneous matrix without z translation: ";
            yDebug() << transform_matrix.toString();
            yDebug() << "Axis size: " << candidate.ax_size.toString();
            transform_matrix.setSubcol(transform_matrix.subcol(0,3,3) - z_offset*gz/norm(gz), 0, 3);
            yDebug() << "Homogeneous matrix with z translation: ";
            yDebug() << transform_matrix.toString();

            Vector gy = candidate.ax_orientation.getCol(1);
            Vector rootz = Vector(3, 0.0);
            rootz(2) = 1.0;

            //  filter candidates
            if ((transform_matrix(2,3) > TABLE_HEIGHT_Z)  && (dot(rootz, gy) <= 0.0))
            {
                //  set the vtk transform to this 4x4 transformation matrix
                candidate.ax_transform->SetMatrix(yarpMatToVTKMat(transform_matrix));
                candidate.ax_actor->SetUserTransform(candidate.ax_transform);

                //  add the axis triad as actor for the rendering
                candidate.ax_actor->AxisLabelsOff();
                candidate.ax_actor->SetTotalLength(0.02, 0.02, 0.02);

                //  save the good candidates and add them to render
                filtered_candidates.push_back(candidate);
                vtk_renderer->AddActor(candidate.ax_actor);
            }
        }

        pose_candidates = filtered_candidates;

        yDebug() << "refreshing grasp computation";
        yDebug() << "Props rendered: " << vtk_renderer->GetNumberOfPropsRendered();

        return;

    }

    /****************************************************************/
    vtkSmartPointer<vtkMatrix4x4> yarpMatToVTKMat (const Matrix m_yarp)
    {
        vtkSmartPointer<vtkMatrix4x4> m_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
        m_vtk->Zero();
        for (size_t i = 0; i<4; i++)
        {
            for (size_t j=0; j<4; j++)
            {
                m_vtk->SetElement(i, j, m_yarp(i,j));
            }
        }
        return m_vtk;
    }

public:

    //  set up the constructor
    DisplaySuperQ(): TABLE_HEIGHT_Z(-0.148), RateThread(100000), closing(false), PCproc(this), SQprocAG(this, SuperquadricType::ANALYTICAL_GRAD_SQ),
        SQprocFD(this, SuperquadricType::FINITE_DIFF_SQ) {}
};

/****************************************************************/

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

    DisplaySuperQ disp;

    return disp.runModule(rf);
}
