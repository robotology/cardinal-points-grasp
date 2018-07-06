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
#include <algorithm>
#include <memory>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/math/Math.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/PolyDriver.h>

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
using namespace yarp::dev;

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
    RpcClient reach_calib_rpc;
    RpcClient table_calib_rpc;
    RpcServer module_rpc;   //will be replaced by idl services

    bool closing;

    string hand;
    string robot;
    WhichHand grasping_hand;

    //  client for cartesian interface
    PolyDriver left_arm_client, right_arm_client;
    ICartesianControl *icart;

    //  backup context
    int context_backup;

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
    vector<shared_ptr<GraspPose>> pose_candidates;

    //  grasp pose actors (temporary fix)
    vector<vtkSmartPointer<vtkAxesActor>> pose_actors;
    vector<vtkSmartPointer<vtkCaptionActor2D>> pose_captions;

    //  filtering constants
    double table_height_z;
    double palm_width;
    double grasp_diameter;
    double finger_length;

    //  visualization parameters
    int x, y, h, w;

    bool configure(ResourceFinder &rf) override
    {

        moduleName = rf.check("name", Value("graspProcessor")).toString();
        robot = (rf.check("sim")? "icubSim" : "icub");
        x = rf.check("x", Value(0)).asInt();
        y = rf.check("y", Value(0)).asInt();
        w = rf.check("width", Value(600)).asInt();
        h = rf.check("height", Value(600)).asInt();

        Property optionLeftArm, optionRightArm;

        optionLeftArm.put("device", "cartesiancontrollerclient");
        optionLeftArm.put("remote", "/" + robot + "/cartesianController/left_arm");
        optionLeftArm.put("local", "/" + moduleName + "/cartesianClient/left_arm");

        optionRightArm.put("device", "cartesiancontrollerclient");
        optionRightArm.put("remote", "/" + robot + "/cartesianController/right_arm");
        optionRightArm.put("local", "/" + moduleName + "/cartesianClient/right_arm");

        //  open the necessary ports
        superq_rpc.open("/" + moduleName + "/superquadricRetrieve:rpc");
        point_cloud_rpc.open("/" + moduleName + "/pointCloud:rpc");
        action_render_rpc.open("/" + moduleName + "/actionRenderer:rpc");
        reach_calib_rpc.open("/" + moduleName + "/reachingCalibration:rpc");
        table_calib_rpc.open("/" + moduleName + "/tableCalib:rpc");
        module_rpc.open("/" + moduleName + "/cmd:rpc");

        //  open client and view
        if(!(left_arm_client.open(optionLeftArm) && right_arm_client.open(optionRightArm)))
            return false;

        //  attach callback
        attach(module_rpc);

        //  initialize an empty point cloud to display
        PointCloud<DataXYZRGBA> pc;
        pc.clear();
        vtk_points = unique_ptr<Points>(new Points(pc, 3));

        //  initialize a zero-superquadric to display
        double r[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 1};
        vtk_superquadric = unique_ptr<Superquadric>(new Superquadric(Vector(12, r), 1.2));

        //  set up rendering window and interactor
        vtk_renderer = vtkSmartPointer<vtkRenderer>::New();
        vtk_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        vtk_renderWindow->SetSize(w, h);
        vtk_renderWindow->SetPosition(x, y);
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

        vtk_camera = vtkSmartPointer<vtkCamera>::New();
        vtk_camera->SetPosition(0.1, 0.0, 0.5);
        vtk_camera->SetViewUp(0.0, 0.0, 1.0);
        vtk_renderer->SetActiveCamera(vtk_camera);

        //  prepare the pose actors vector
        for (size_t idx = 0; idx < 25; idx++)
        {
            vtkSmartPointer<vtkAxesActor> ax_actor = vtkSmartPointer<vtkAxesActor>::New();
            vtkSmartPointer<vtkCaptionActor2D> cap_actor = vtkSmartPointer<vtkCaptionActor2D>::New();
            ax_actor->VisibilityOff();
            cap_actor->VisibilityOff();
            pose_actors.push_back(ax_actor);
            pose_captions.push_back(cap_actor);
            vtk_renderer->AddActor(pose_actors[idx]);
            vtk_renderer->AddActor(pose_captions[idx]);
        }

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
        reach_calib_rpc.interrupt();
        module_rpc.interrupt();
        table_calib_rpc.interrupt();
        closing = true;

        return true;
    }

    /****************************************************************/
    bool close() override
    {
        superq_rpc.close();
        point_cloud_rpc.close();
        action_render_rpc.close();
        reach_calib_rpc.close();
        module_rpc.close();
        left_arm_client.close();
        right_arm_client.close();
        table_calib_rpc.close();

        return true;
    }

    /****************************************************************/
    bool respond(const Bottle& command, Bottle& reply) override
    {
        //  parse for available commands

        bool cmd_success = false;
        Vector grasp_pose(7, 0.0);
        string obj;
        string hand;

        if (command.get(0).toString() == "grasp_pose")
        {
            //  normal operation
            if (command.size() == 3)
            {
                obj = command.get(1).toString();
                hand = command.get(2).toString();
                if (hand == "right")
                {
                    grasping_hand = WhichHand::HAND_RIGHT;
                }
                else if (hand == "left")
                {
                    grasping_hand = WhichHand::HAND_LEFT;
                }
                else
                {
                    reply.addVocab(Vocab::encode("nack"));
                    return true;
                }
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
            PointCloud<DataXYZRGBA> pc;
            yDebug() << "Requested object: " << obj;          
            if (requestRefreshPointCloud(pc, obj))
            {
                if (requestRefreshSuperquadric(pc))
                {
                    cmd_success = computeGraspPose(grasp_pose);
                    yInfo() << "Pose retrieved: " << grasp_pose.toString();
                }
            }
        }

        if (command.get(0).toString() == "grasp")
        {
            //  obtain grasp and render it
            if (command.size() == 3)
            {
                obj = command.get(1).toString();
                hand = command.get(2).toString();
                if (hand == "right")
                {
                    grasping_hand = WhichHand::HAND_RIGHT;
                }
                else if (hand == "left")
                {
                    grasping_hand = WhichHand::HAND_LEFT;
                }
                else
                {
                    reply.addVocab(Vocab::encode("nack"));
                    return true;
                }
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
            PointCloud<DataXYZRGBA> pc;
            yDebug() << "Requested object: " << obj;
            if (requestRefreshPointCloud(pc, obj))
            {
                if (requestRefreshSuperquadric(pc))
                {
                    if (computeGraspPose(grasp_pose))
                    {
                        yInfo() << "Pose retrieved: " << grasp_pose.toString();
                        cmd_success = executeGrasp(grasp_pose);
                    }
                }
            }
        }

        if (command.get(0).toString() == "from_off_file")
        {
            //  process point cloud from file and perform candidate ranking
            if (command.size() == 3)
            {
                obj = command.get(1).toString();
                hand = command.get(2).toString();
                if (hand == "right")
                {
                    grasping_hand = WhichHand::HAND_RIGHT;
                }
                else if (hand == "left")
                {
                    grasping_hand = WhichHand::HAND_LEFT;
                }
                else
                {
                    reply.addVocab(Vocab::encode("nack"));
                    return true;
                }
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
            string filename = command.get(1).toString();
            //  load point cloud from .off file, store it and refresh point cloud, request and refresh superquadric and compute poses
            PointCloud<DataXYZRGBA> pc;

            ifstream file(filename.c_str());
            string delimiter = " ";

            if (!file.is_open())
            {
                yError() << "Unable to open file";
                reply.addVocab(Vocab::encode("nack"));
                return false;
            }

            //  parse the OFF file line by line
            string line;
            getline(file, line);
            if (line != "COFF")
            {
                yError() << "File parsing failed";
                reply.addVocab(Vocab::encode("nack"));
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
                pc(idx).r = static_cast<unsigned char>(stoi(parsed_line[3]));
                pc(idx).g = static_cast<unsigned char>(stoi(parsed_line[4]));
                pc(idx).b = static_cast<unsigned char>(stoi(parsed_line[5]));
                pc(idx).a = 255;
            }

            if (pc.size() > 0)
            {
                refreshPointCloud(pc);
                if (requestRefreshSuperquadric(pc))
                {
                    cmd_success = computeGraspPose(grasp_pose);
                    yInfo() << "Pose retrieved: " << grasp_pose.toString();
                }
            }
        }

        reply.addVocab(Vocab::encode(cmd_success ? "ack":"nack"));

        return true;

    }

    /****************************************************************/
    void setGraspContext()
    {
        //  set up the context for the grasping planning and execution
        //  enable all joints
        Vector dof;
        icart->getDOF(dof);
        //yDebug() << "Previous DOF config: [" << dof.toString() << "]";
        Vector new_dof(10, 1);
        new_dof(1) = 0.0;
        icart->setDOF(new_dof, dof);
        //yDebug() << "New DOF config: [" << new_dof.toString() << "]";
        icart->setPosePriority("position");
        icart->setInTargetTol(0.001);

        //  display and set motion limits
//        double min_torso_pitch, max_torso_pitch;
//        double min_torso_yaw, max_torso_yaw;
//        double min_torso_roll, max_torso_roll;
//        icart -> getLimits(0, &min_torso_pitch, &max_torso_pitch);
//        icart -> getLimits(1, &min_torso_roll, &max_torso_roll);
//        icart -> getLimits(2, &min_torso_yaw, &max_torso_yaw);
//        yInfo() << "Torso current pitch limits: min " << min_torso_pitch << " max " << max_torso_pitch;
//        yInfo() << "Torso current roll limits: min " << min_torso_roll << " max " << max_torso_roll;
//        yInfo() << "Torso current yaw limits: min " << min_torso_yaw << " max " << max_torso_yaw;
//        icart -> setLimits(0, min_torso_pitch, max_torso_pitch);
//        icart -> setLimits(1, min_torso_roll, max_torso_roll);
//        icart -> setLimits(2, min_torso_yaw, max_torso_yaw);

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

           //   position the camera to look at point cloud
           vector<double> bounds(6), centroid(3);
           vtk_points->get_polydata()->GetBounds(bounds.data());

           for (size_t i=0; i<centroid.size(); i++)
           {
               centroid[i] = 0.5 * (bounds[i<<1] + bounds[(i<<1)+1]);
           }

           vtk_camera->SetPosition(centroid[0]+0.5, centroid[1], centroid[2]+1.0);
           vtk_camera->SetViewUp(0, 0, 1);
           vtk_camera->SetFocalPoint(centroid.data());
       }
    }

    /****************************************************************/
    void refreshSuperquadric(const Vector &superq_params)
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
        Bottle cmd_reply;

        cmd_request.addString("look");
        cmd_request.addString(object);
        cmd_request.addString("wait");

        action_render_rpc.write(cmd_request, cmd_reply);
        if (cmd_reply.toString() != "[ack]")
        {
            yError() << "Didn't manage to look at the object";
            return false;
        }

        point_cloud.clear();
        cmd_request.clear();
        cmd_reply.clear();

        cmd_request.addString("get_point_cloud");
        cmd_request.addString(object);

        point_cloud_rpc.write(cmd_request, cmd_reply);

        //  cheap workaround to get the point cloud
        Bottle* pcBt = cmd_reply.get(0).asList();
        bool success = point_cloud.fromBottle(*pcBt);

        if (success && (point_cloud.size() > 0))
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
    bool isCandidateGraspFeasible(shared_ptr<GraspPose> &candidate_pose)
    {
        //  filter candidate grasp. True for good grasp
        Vector root_z_axis(3, 0.0);
        root_z_axis(2) = 1;

        using namespace yarp::math;

        /*
         * Filtering parameters:
         * 1 - sufficient height wrt table
         * 2 - grasp width wrt hand x axis
         * 3 - sufficient size wrt palm width
         * 4 - thumb cannot point down
         */

        bool ok1, ok2, ok3, ok4;
        ok1 = candidate_pose->pose_translation(2) - palm_width/2 > table_height_z;
        ok2 = candidate_pose->pose_ax_size(0) * 2 < grasp_diameter;
        ok3 = candidate_pose->pose_ax_size(1) * 2 > palm_width/2;
        ok4 = dot(candidate_pose->pose_rotation.getCol(1), root_z_axis) <= 0.1;

        return (ok1 && ok2 && ok3 && ok4);
    }

    /****************************************************************/
    void getPoseCostFunction(shared_ptr<GraspPose> &candidate_pose)
    {
        //  compute precision for movement
        using namespace yarp::math;

        Vector x_d = candidate_pose->pose_translation;
        Vector o_d = dcm2axis(candidate_pose->pose_rotation);
        Vector x_d_hat, o_d_hat, q_d_hat;

        icart->askForPose(x_d, o_d, x_d_hat, o_d_hat, q_d_hat);

        yDebug() << "Requested: " << candidate_pose->pose_transform.toString();

        //  calculate position cost function (first component of cost function)
        candidate_pose->pose_cost_function(0) = norm(x_d - x_d_hat);

        //  calculate orientation cost function
        Matrix tmp = axis2dcm(o_d_hat).submatrix(0,2, 0,2);
        Matrix orientation_error_matrix =  candidate_pose->pose_rotation * tmp.transposed();
        Vector orientation_error_vector = dcm2axis(orientation_error_matrix);

        candidate_pose->pose_cost_function(1) = norm(orientation_error_vector.subVector(0,2) * sin(orientation_error_vector(3))) / norm(orientation_error_vector.subVector(0,2));


        candidate_pose->pose_cost_function(1) = 0.5*candidate_pose->pose_cost_function(1) + 0.5*(1-candidate_pose->pose_ax_size(1)/yarp::math::findMax(candidate_pose->pose_ax_size));

        yDebug() << "Cost function: " << candidate_pose->pose_cost_function.toString();

    }

    /****************************************************************/
    void computeGraspCandidates()
    {
        //  compute a series of viable grasp candidates according to superquadric parameters
        LockGuard lg(mutex);

        if (grasping_hand == WhichHand::HAND_LEFT)
        {
            left_arm_client.view(icart);
        }
        else if (grasping_hand == WhichHand::HAND_RIGHT)
        {
            right_arm_client.view(icart);
        }
        else
        {
            yError() << "Invalid arm!";
            return;
        }

        //  retrieve table height
        //  otherwise, leave default value
        if (robot != "icubSim" && table_calib_rpc.getOutputCount() > 0)
        {
            Bottle table_cmd, table_rply;
            table_cmd.addString("get");
            table_cmd.addString("table");

            table_calib_rpc.write(table_cmd, table_rply);
            table_height_z = table_rply.get(0).asDouble();
        }
        else
        {
            yInfo() << "Unable to retrieve table height, using default.";
        }

        //  store the context for the previous iKinCartesianController config
        icart->storeContext(&context_backup);

        //  set up the context for the computation of the candidates
        setGraspContext();

        //  detach vtk actors corresponding to poses, if any are present
//        for (auto &grasp_pose : pose_candidates)
//        {
//            vtk_renderer->RemoveActor(grasp_pose->pose_vtk_actor);
//            vtk_renderer->RemoveActor(grasp_pose->pose_vtk_caption_actor);
//        }

        //  set pose actors to be invisible
        for (auto &pose_actor : pose_actors)
        {
            pose_actor->VisibilityOff();
        }

        for (auto &cap_actor : pose_captions)
        {
            cap_actor->VisibilityOff();
        }


        //  get superquadric parameters
        pose_candidates.clear();
        Vector superq_center = vtk_superquadric->getCenter();
        Vector superq_XYZW_orientation = vtk_superquadric->getOrientationXYZW();
        Vector superq_axes_size = vtk_superquadric->getAxesSize();

        //  get orientation of the superq in 3x3 rotation matrix form
        using namespace yarp::math;
        superq_XYZW_orientation(3) /= (180.0 / M_PI);
        Matrix superq_mat_orientation = axis2dcm(superq_XYZW_orientation).submatrix(0,2, 0,2);

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
                    shared_ptr<GraspPose> candidate_pose = shared_ptr<GraspPose>(new GraspPose);
                    candidate_pose->pose_rotation.setCol(0, gx);
                    candidate_pose->pose_rotation.setCol(1, gy);
                    candidate_pose->pose_rotation.setCol(2, gz);

                    //  we can either have side or top grasps.
                    //  it is helpful to have a variable to discriminate the grasp
                    Vector z_root(3, 0.0);
                    z_root(2) = 1.0;
                    bool is_side_grasp = (dot(-1*z_root, gy) > 0.9);

                    //  set the superquadric size in the direction of each g_axis
                    candidate_pose->pose_ax_size(0) = superq_axes_size(idx/2);
                    candidate_pose->pose_ax_size(1) = superq_axes_size(jdx/2);
                    candidate_pose->pose_ax_size(2) = superq_axes_size(3 - idx/2 - jdx/2);

                    //  translate the pose along gz and gx according to the palm size
                    //  rotate the pose around the hand y axis
                    //  sign of operations depends upon the hand we are using
                    //  the placement of the hand wrt the superquadric center depends on the size along gx

                    if (grasping_hand == WhichHand::HAND_RIGHT)
                    {
                        double angle = -M_PI/4;
                        Vector y_rotation_transform(4, 0.0);
                        y_rotation_transform(1) = 1.0;
                        y_rotation_transform(3) = angle;
                        candidate_pose->pose_rotation = candidate_pose->pose_rotation * yarp::math::axis2dcm(y_rotation_transform).submatrix(0, 2, 0, 2);
                        candidate_pose->pose_translation = superq_center - ((candidate_pose->pose_ax_size(2) + 0.02) * gz/norm(gz)) - ((0*candidate_pose->pose_ax_size(0) + 0.02) * gx/norm(gx));
                    }
                    else
                    {
                        double angle = M_PI/4;
                        Vector y_rotation_transform(4, 0.0);
                        y_rotation_transform(1) = 1.0;
                        y_rotation_transform(3) = angle;
                        candidate_pose->pose_rotation = candidate_pose->pose_rotation * yarp::math::axis2dcm(y_rotation_transform).submatrix(0, 2, 0, 2);
                        candidate_pose->pose_translation = superq_center + ((candidate_pose->pose_ax_size(2) + 0.02) * gz/norm(gz)) - ((0*candidate_pose->pose_ax_size(0) + 0.02) * gx/norm(gx));
                    }

                    if (isCandidateGraspFeasible(candidate_pose))
                    {
                        //  if the grasp is close to the table surface, we need to adjust the pose to avoid collision
                        bool side_low = is_side_grasp && ((candidate_pose->pose_translation(2) - palm_width) < table_height_z);
                        bool top_low = !is_side_grasp && ((candidate_pose->pose_translation(2) - finger_length) < table_height_z);

                        if (side_low)
                        {
                            //  lift up the grasp closer to the upper end of the superquadric
                            candidate_pose->pose_translation(2) = superq_center(2) + candidate_pose->pose_ax_size(1) - (palm_width * 2/3);
                        }
                        if (top_low)
                        {
                            //  grab the object with the grasp center on top of the superquadric center
                            candidate_pose->pose_translation(2) = superq_center(2) - superq_axes_size(2) + finger_length;
                        }

                        if (!candidate_pose->setHomogeneousTransform(candidate_pose->pose_rotation, candidate_pose->pose_translation))
                        {
                            yError() << "Error setting homogeneous transform!";
                            continue;
                        }

                        //  if candidate is good, set vtk transform and actor
                        candidate_pose->setvtkTransform(candidate_pose->pose_transform);
                        candidate_pose->pose_vtk_actor->SetUserTransform(candidate_pose->pose_vtk_transform);
                        pose_actors[idx*search_space_gy.size() + jdx]->SetUserTransform(candidate_pose->pose_vtk_transform);

                        //  calculate the cost function for the pose
                        getPoseCostFunction(candidate_pose);

                        //  fix graphical properties
//                        candidate_pose->pose_vtk_actor->AxisLabelsOff();
//                        candidate_pose->pose_vtk_actor->SetTotalLength(0.02, 0.02, 0.02);

                        stringstream ss;
                        ss << fixed << setprecision(3) << candidate_pose->pose_cost_function(0) << "_" << fixed << setprecision(3) << candidate_pose->pose_cost_function(1);
                        candidate_pose->setvtkActorCaption(ss.str());

                        candidate_pose->pose_vtk_actor->ShallowCopy(pose_actors[idx*search_space_gy.size() + jdx]);
                        pose_actors[idx*search_space_gy.size() + jdx]->AxisLabelsOff();
                        pose_actors[idx*search_space_gy.size() + jdx]->SetTotalLength(0.02, 0.02, 0.02);
                        pose_actors[idx*search_space_gy.size() + jdx]->VisibilityOn();

                        pose_captions[idx*search_space_gy.size() + jdx]->VisibilityOn();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetTextActor()->SetTextScaleModeToNone();
                        pose_captions[idx*search_space_gy.size() + jdx]->SetCaption(candidate_pose->pose_vtk_caption_actor->GetCaption());
                        pose_captions[idx*search_space_gy.size() + jdx]->BorderOff();
                        pose_captions[idx*search_space_gy.size() + jdx]->LeaderOn();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->SetFontSize(20);
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->FrameOff();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->ShadowOff();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->BoldOff();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->ItalicOff();
                        pose_captions[idx*search_space_gy.size() + jdx]->GetCaptionTextProperty()->SetColor(1.0, 1.0, 1.0);
                        pose_captions[idx*search_space_gy.size() + jdx]->SetAttachmentPoint(candidate_pose->pose_vtk_caption_actor->GetAttachmentPoint());

                        //  add actor to renderer
                        //vtk_renderer->AddActor(candidate_pose->pose_vtk_actor);
                        //vtk_renderer->AddActor(candidate_pose->pose_vtk_caption_actor);
                        pose_candidates.push_back(candidate_pose);
                    }
                }
            }
        }

        //  restore previous context
        icart->restoreContext(context_backup);

        return;

    }

    /****************************************************************/
    bool getBestCandidatePose(shared_ptr<GraspPose> &best_candidate)
    {
        //  compute which is the best pose wrt cost function

        if (pose_candidates.size())
        {
            //  sort poses based on < operator defined for GraspPose
            //  from smallest to largest position error
            sort(pose_candidates.begin(), pose_candidates.end());

            //  select candidate with the best orientation precision
            //  precision of less than 1 cm is not accepted
            if (pose_candidates[0]->pose_cost_function(0) < 0.01)
            {
                best_candidate = pose_candidates[0];
                for(size_t idx = 1; (idx < pose_candidates.size()) && (pose_candidates[idx]->pose_cost_function(0) < 0.01); idx++)
                {
                    if (pose_candidates[idx]->pose_cost_function(1) < best_candidate->pose_cost_function(1))
                    {
                        best_candidate = pose_candidates[idx];
                    }
                }
            }
            else
            {
                return false;
            }

            //  display the best candidate
            LockGuard lg(mutex);
            //best_candidate->pose_vtk_actor->SetTotalLength(0.06, 0.06, 0.06);
            for (auto &caption : pose_captions)
            {

                if (caption->GetCaption() == NULL)
                    continue;

                string string2(best_candidate->pose_vtk_caption_actor->GetCaption());
                string string1(caption->GetCaption());

                if (string1 == string2)
                {
                    caption->GetCaptionTextProperty()->SetColor(0.0, 1.0, 0.0);
                    caption->GetCaptionTextProperty()->BoldOn();
                    break;
                }
            }
            yInfo() << "Best candidate: cartesian " << best_candidate->pose_translation.toString()
                    << " pose " << yarp::math::dcm2axis(best_candidate->pose_rotation).toString();
            yInfo() << "Cost: " << best_candidate->pose_cost_function.toString();
            yDebug()<< "candidate ax size: " << best_candidate->pose_ax_size.toString();
            return true;
        }
        else
        {
            return false;
        }

    }

    /****************************************************************/
    bool fixReachingOffset(const Vector &poseToFix, Vector &poseFixed)
    {
        //  fix the pose offset accordint to iolReachingCalibration
        //  pose is supposed to be (x y z gx gy gz theta)
        Bottle command, reply;

        command.addString("get_location_nolook");
        if (grasping_hand == WhichHand::HAND_LEFT)
        {
            command.addString("iol-left");
        }
        else
        {
            command.addString("iol-right");
        }

        command.addDouble(poseToFix(0));    //  x value
        command.addDouble(poseToFix(1));    //  y value
        command.addDouble(poseToFix(2));    //  z value

        reach_calib_rpc.write(command, reply);

        //  incoming reply is going to be (success x y z)
        if (reply.get(0).asString() == "ok")
        {
            poseFixed = poseToFix;
            poseFixed(0) = reply.get(1).asDouble();
            poseFixed(1) = reply.get(2).asDouble();
            poseFixed(2) = reply.get(3).asDouble();
            return true;
        }
        else
        {
            yError() << "Failure retrieving fixed pose";
            return false;
        }
    }

    /****************************************************************/
    bool computeGraspPose(Vector &pose)
    {
        //  execute the pose computation pipeline
        bool success = false;

        //  if anything goes wrong, the operation fails and the function
        //  returns a failure
        computeGraspCandidates();
        shared_ptr<GraspPose> best_pose;
        if(getBestCandidatePose(best_pose))
        {
            if (robot == "icub")
            {
                success = fixReachingOffset(best_pose->getPose(), pose);
            }
            else
            {
                //  no need to fix the offset in simulation
                pose = best_pose->getPose();
                success = true;
            }
        }

        return success;
    }

    /****************************************************************/
    bool executeGrasp(Vector &pose)
    {
        if (robot == "icub")
        {
            //  communication with actionRenderingEngine/cmd:io
            //  grasp("cartesian" x y z gx gy gz theta) ("approach" (-0.05 0 +-0.05 0.0)) "left"/"right"
            Bottle command, reply;

            command.addString("grasp");
            Bottle &ptr = command.addList();
            ptr.addString("cartesian");
            ptr.addDouble(pose(0));
            ptr.addDouble(pose(1));
            ptr.addDouble(pose(2));
            ptr.addDouble(pose(3));
            ptr.addDouble(pose(4));
            ptr.addDouble(pose(5));
            ptr.addDouble(pose(6));


            Bottle &ptr1 = command.addList();
            ptr1.addString("approach");
            Bottle &ptr2 = ptr1.addList();
            ptr2.addDouble(-0.05);
            ptr2.addDouble(0.0);
            if (grasping_hand == WhichHand::HAND_LEFT)
            {
                ptr2.addDouble(0.05);
                command.addString("left");
            }
            else
            {
                ptr2.addDouble(-0.05);
                command.addString("right");
            }
            ptr2.addDouble(0.0);

            yInfo() << command.toString();
            action_render_rpc.write(command, reply);
            if (reply.toString() == "[ack]")
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            //  simulation context, suppose there is no actionsRenderingEngine running
            setGraspContext();
            Vector previous_x(3), previous_o(4);
            icart->getPose(previous_x, previous_o);
            icart->goToPoseSync(pose.subVector(0, 2), pose.subVector(3,6));
            icart->waitMotionDone();
            icart->goToPoseSync(previous_x, previous_o);
            icart->waitMotionDone();
            icart->restoreContext(context_backup);
            return true;
        }
    }


public:
    GraspProcessorModule(): closing(false), table_height_z(-0.15), palm_width(0.08),
        finger_length(0.08), grasp_diameter(0.08), grasping_hand(WhichHand::HAND_RIGHT)  {}

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
