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
    bool setHomogeneousTransform(const Matrix &rotation, const Vector &translation)
    {
        //  set the 4x4 homogeneous transform given 3x3 rotation and 1x3 translation
        if (rotation.cols() == 3 && rotation.rows() == 3 && translation.size() == 3)
        {
            pose_transform.setSubmatrix(rotation, 0, 0);
            pose_transform.setSubcol(translation, 0, 3);
            return true;
        }
        else
            return false;
    }

    void setvtkTransform(const Matrix &transform)
    {
        vtkSmartPointer<vtkMatrix4x4> m_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
        m_vtk->Zero();
        for (size_t i = 0; i < 4; i++)
        {
            for(size_t j = 0; j < 4; j++)
            {
                m_vtk->SetElement(i, j, transform(i, j));
            }
        }

        pose_vtk_transform->SetMatrix(m_vtk);
    }


};

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
                 void *vtkNotUsed(callData)) override
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

enum class WhichHand
{
    HAND_RIGHT,
    HAND_LEFT
};

/****************************************************************/

class GraspProcessorModule : public RFModule
{
    string moduleName;

    RpcClient superq_rpc;
    RpcClient point_cloud_rpc;
    RpcClient action_render_rpc;    //not used atm
    RpcServer module_rpc;   //will be replaced by idl services

    bool closing;

    WhichHand grasping_hand;

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
//        if (rf.check("name"))
//        {
//            moduleName = rf.find("name").asString();
//        }
//        else
//        {
//            moduleName = "graspProcessor";
//        }
        moduleName = rf.check("name", Value("graspProcessor")).asString();

        //  parse for grasping hand
        if (rf.check("hand"))
        {
            if (rf.find("hand").asString() == "right")
            {
                grasping_hand = WhichHand::HAND_RIGHT;
            }
            else if (rf.find("hand").asString() == "left")
            {
                grasping_hand = WhichHand::HAND_LEFT;
            }
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
        //  parse for available commands

        bool cmd_success = false;

        if (command.check("grasp_pose"))
        {
            //  normal operation
            string obj = command.find("grasp_pose").asString();
            PointCloud<DataXYZRGBA> pc;
            yDebug() << "Requested object: " << obj;
            if (requestRefreshPointCloud(pc, obj))
            {
                if (requestRefreshSuperquadric(pc))
                {
                    computeGraspCandidates();
                    getBestCandidatePose();
                    cmd_success = true;
                }
            }
        }

        if (command.check("from_off_file"))
        {
            //  process point cloud from file and perform candidate ranking
            string filename = command.find("from_off_file").asString();
            //  load point cloud from .off file, store it and refresh point cloud, request and refresh superquadric and compute poses
            PointCloud<DataXYZRGBA> pc;

            ifstream file(filename.c_str());
            string delimiter = " ";

            if (!file.is_open())
            {
                yError() << "Unable to open file";
                return false;
            }

            //  parse the OFF file line by line

            string line;
            getline(file, line);
            if (line != "COFF")
            {
                yError() << "File parsing failed";
                return false;
            }
            line.clear();
            getline(file, line);
            size_t pos = 0;
            pos = line.find(delimiter);
            pc.resize(stoul(line.substr(0, pos)));

            //  get a line for each point and build a point cloud
            for (size_t idx = 0; idx < pc.size(); idx++)
            {
                line.clear();
                pos = 0;
                getline(file, line);
                vector<string> parsed_line;
                while ((pos = line.find(delimiter)) != string::npos)
                {
                    parsed_line.push_back(line.substr(0, pos));
                    line.erase(0, pos+delimiter.length());
                }
                parsed_line.push_back(line);
                pc(idx).x = stof(parsed_line[0]);
                pc(idx).y = stof(parsed_line[1]);
                pc(idx).z = stof(parsed_line[2]);
                pc(idx).r = (unsigned char)stoi(parsed_line[3]);
                pc(idx).g = (unsigned char)stoi(parsed_line[4]);
                pc(idx).b = (unsigned char)stoi(parsed_line[5]);
                pc(idx).a = 255;
            }

            if (pc.size() > 0)
            {
                refreshPointCloud(pc);
                requestRefreshSuperquadric(pc);
                computeGraspCandidates();
                getBestCandidatePose();
            }
        }

        reply.addVocab(Vocab::encode(cmd_success ? "ack":"nack"));
        return true;

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

        Vector superq_params_sorted(12, 0.0);

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
    bool requestRefreshPointCloud(PointCloud<DataXYZRGBA> &point_cloud, const string &object)
    {
        //  query point-cloud-read via rpc for the point cloud
        //  command: get_point_cloud objectName
        //  put point cloud into container, return true if operation was ok
        //  or call refreshpointcloud
        Bottle cmd_request;
        cmd_request.clear();
        Bottle cmd_reply;
        cmd_reply.clear();

        point_cloud.clear();

        cmd_request.addString("get_point_cloud");
        cmd_request.addString(object);

        point_cloud_rpc.write(cmd_request, cmd_reply);

        //  cheap workaround to get the point cloud
        Bottle* pcBt = cmd_reply.get(0).asList();
        bool success = point_cloud.fromBottle(*pcBt);

        if (success && (point_cloud.size() > 0))
        {
            yDebug() << "Point cloud retrieved; contains " << point_cloud.size() << "points";
//            for (size_t idx = 0; idx < point_cloud.size(); idx++)
//            {
//                yDebug() << "Point " << idx << ":" << point_cloud(idx).x << point_cloud(idx).y << point_cloud(idx).z;
//            }
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
    bool isCandidateGraspFeasible(const GraspPose &candidate_pose)
    {
        //  filter candidate grasp. True for good grasp
        Vector root_z_axis(3, 0.0);
        root_z_axis(2) = 1;

        using namespace yarp::math;

        /*
         * Filtering parameters:
         * 1 - sufficient height wrt table
         * 2 - grasp width wrt hand x axis
         * 3 - palm width
         * 4 - thumb cannot point down
         */

        bool ok1, ok2, ok3, ok4;
        ok1 = candidate_pose.pose_transform(3, 2) > table_height_z + palm_width_y/2;
        ok2 = candidate_pose.pose_ax_size(0) * 2 < grasp_width_x;
        ok3 = candidate_pose.pose_ax_size(1) * 2 < palm_width_y;
        ok4 = dot(candidate_pose.pose_transform.subcol(0, 1, 3), root_z_axis) <= 0.3;

        return (ok1 && ok2 && ok3 && ok4);
    }

    /****************************************************************/
    void computeGraspCandidates()
    {
        //  compute a series of viable grasp candidates according to superquadric parameters
        LockGuard lg(mutex);

        //  detach vtk actors corresponding to poses, if any are present
        for (GraspPose grasp_pose : pose_candidates)
        {
            vtk_renderer->RemoveActor(grasp_pose.pose_vtk_actor);
        }

        //  get superquadric parameters
        pose_candidates.clear();
        Vector superq_center = vtk_superquadric->getCenter();
        Vector superq_XYZW_orientation = vtk_superquadric->getOrientationXYZW();
        Vector superq_axes_size = vtk_superquadric->getAxesSize();

        //  get orientation of the superq in 3x3 rotation matrix form
        using namespace yarp::math;
        superq_XYZW_orientation(3) /= (180 / M_PI);
        Matrix superq_mat_orientation = axis2dcm(superq_XYZW_orientation).submatrix(0,2, 0,2);

        yDebug() << "Superquadric orientation: " << superq_mat_orientation.toString();

        //  columns of rotation matrix are superq axes direction
        //  in root reference frame;
        Vector sq_axis_x = superq_mat_orientation.getCol(0);
        Vector sq_axis_y = superq_mat_orientation.getCol(1);
        Vector sq_axis_z = superq_mat_orientation.getCol(2);

        //  create all possible candidates for pose evaluation
        //  create search space for grasp axes x and y
        vector<Vector> search_space_gx = {sq_axis_x, -1*sq_axis_x, sq_axis_y, -1*sq_axis_y};
        vector<Vector> search_space_gy = {sq_axis_x, -1*sq_axis_x, sq_axis_y, -1*sq_axis_y, sq_axis_z, -1*sq_axis_z};

        //  create actual candidates
        for (size_t idx = 0; idx < search_space_gx.size(); idx++)
        {
            Vector gx = search_space_gx[idx];
            //  for each candidate gx axis, try all gy possibilities
            for (size_t jdx = 0; jdx < search_space_gy.size(); jdx++)
            {
                //  create a gx, gy orthogonal couple
                Vector gy = search_space_gy[jdx];
                if (dot(gx, gy)*dot(gx, gy) < 0.0001)
                {
                    //  create gz with cross product
                    //  create candidate entry
                    Vector gz = cross(gx, gy);
                    GraspPose candidate_pose;
                    candidate_pose.pose_rotation.setCol(0, gx);
                    candidate_pose.pose_rotation.setCol(1, gy);
                    candidate_pose.pose_rotation.setCol(2, gz);

                    //  set the superquadric size in the direction of each g_axis
                    candidate_pose.pose_ax_size(0) = superq_axes_size(idx/2);
                    candidate_pose.pose_ax_size(1) = superq_axes_size(jdx/2);
                    candidate_pose.pose_ax_size(2) = superq_axes_size(3 - idx/2 - jdx/2);

                    //  translate along gz according to superquadric size
                    //  minus sign, since we are using right hand
                    //  left hand would have plus sign
                    if (grasping_hand == WhichHand::HAND_RIGHT)
                    {
                        candidate_pose.pose_translation = superq_center - candidate_pose.pose_ax_size(2) * gz/norm(gz);
                    }
                    else if (grasping_hand == WhichHand::HAND_LEFT)
                    {
                        candidate_pose.pose_translation = superq_center + candidate_pose.pose_ax_size(2) * gz/norm(gz);
                    }

                    if (!candidate_pose.setHomogeneousTransform(candidate_pose.pose_rotation, candidate_pose.pose_translation))
                    {
                        yError() << "Error setting homogeneous transform!";
                        continue;
                    }

                    if (isCandidateGraspFeasible(candidate_pose))
                    {
                        //  if candidate is good, set vtk transform and actor
                        candidate_pose.setvtkTransform(candidate_pose.pose_transform);
                        candidate_pose.pose_vtk_actor->SetUserTransform(candidate_pose.pose_vtk_transform);

                        //  fix graphical properties
                        candidate_pose.pose_vtk_actor->AxisLabelsOff();
                        candidate_pose.pose_vtk_actor->SetTotalLength(0.02, 0.02, 0.02);

                        //  add actor to renderer
                        vtk_renderer->AddActor(candidate_pose.pose_vtk_actor);
                        pose_candidates.push_back(candidate_pose);
                    }
                }
            }
        }

        yInfo() << "Feasible grasp candidates computed: " << pose_candidates.size();
        yInfo() << "Object size: x " << 2*superq_axes_size(0) << " y " << 2*superq_axes_size(1) << " z " << 2*superq_axes_size(2);

        return;

    }

    GraspPose getBestCandidatePose()
    {
        //  compute which is the best
        //  bogus lol
        GraspPose best_candidate;

        if (pose_candidates.size())
        {
            best_candidate = pose_candidates[0];
            yInfo() << "Best candidate: cartesian " << best_candidate.pose_translation.toString() << " pose " << yarp::math::dcm2axis(best_candidate.pose_rotation).toString();
        }
        return best_candidate;

    }

public:
    GraspProcessorModule(): closing(false), table_height_z(-0.14), palm_width_y(0.1), grasp_width_x(0.1), grasping_hand(WhichHand::HAND_RIGHT)  {}

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
