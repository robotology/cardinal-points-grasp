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
#include <atomic>

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
using namespace yarp::math;

Mutex mutex;

/****************************************************************/

string prettyError(const char* func_name, const string &message)
{
    //  Nice formatting for errors
    stringstream error;
    error << "[" << func_name << "] " << message;
    return error.str();
}

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
    RpcClient action_render_rpc;
    RpcClient reach_calib_rpc;
    RpcClient table_calib_rpc;
    RpcServer module_rpc;   //will be replaced by idl services

    bool closing;
    std::atomic<bool> halt_requested;

    string robot;
    WhichHand grasping_hand;

    //  client for cartesian interface (for use with the iCub)
    PolyDriver left_arm_client, right_arm_client;
    ICartesianControl *icart;

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

    //  Robot specific parameters
    Vector planar_obstacle; // plane to avoid, typically a table (format (a b c d) following plane equation a.x+b.y+c.z+d=0)
    Vector grasper_bounding_box_right; // bounding box of the right grasper (x_min x_max y_min _y_max z_min z_max) expressed in the robot right grasper frame used by the controller
    Vector grasper_bounding_box_left; // bounding box of the left grasper (x_min x_max y_min _y_max z_min z_max) expressed in the robot left grasper frame used by the controller
    double obstacle_safety_distance; // minimal distance to respect between the grasper and the obstacle
    Vector min_object_size;
    Vector max_object_size;
    Matrix grasper_specific_transform_right;
    Matrix grasper_specific_transform_left;
    Vector grasper_approach_parameters_right;
    Vector grasper_approach_parameters_left;

    // Filtering constants
    double position_error_threshold;

    // Candidate pose generation parameters
    double roundness_threshold; // threshold on the roundness of the object to generate pseudo cardinal poses
    int nb_cardinal_levels; // number of levels of pseudo cardinal poses generated (1 or lower = normal)

    //  visualization parameters
    int x, y, h, w;

    bool configure(ResourceFinder &rf) override
    {
        moduleName = rf.check("name", Value("graspProcessor")).toString();
        if(!rf.check("robot"))
        {
            robot = (rf.check("sim")? "icubSim" : "icub");
        }
        else
        {
            robot = rf.find("robot").asString();
        }

        yInfo() << "Opening module for connection with robot" << robot;

        string control_arms = rf.check("control-arms", Value("both")).toString();
        x = rf.check("x", Value(0)).asInt();
        y = rf.check("y", Value(0)).asInt();
        w = rf.check("width", Value(600)).asInt();
        h = rf.check("height", Value(600)).asInt();

        Bottle *list = rf.find("min_object_size").asList();
        if(list)
        {
            if(list->size() == 3)
            {
                for(int i=0 ; i<3 ; i++) min_object_size[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid min_object_Size dimension in config. Should be 3.");
            }
        }
        else if((robot == "icubSim") || (robot == "icub"))
        {
            min_object_size[1] = 0.04;
        }
        yInfo() << "Grabber specific min graspable object size loaded\n" << min_object_size.toString();

        list = rf.find("max_object_size").asList();
        if(list)
        {
            if(list->size() == 3)
            {
                for(int i=0 ; i<3 ; i++) max_object_size[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid max_object_Size dimension in config. Should be 3.");
            }
        }
        else if((robot == "icubSim") || (robot == "icub"))
        {
            max_object_size[0] = 0.12;
        }
        yInfo() << "Grabber specific max graspable object size loaded\n" << max_object_size.toString();

        Vector grasp_specific_translation(3, 0.0);
        Vector grasp_specific_orientation(4, 0.0);
        list = rf.find("grasp_trsfm_right").asList();
        bool valid_grasp_specific_transform = true;

        if(list)
        {
            if(list->size() == 7)
            {
                for(int i=0 ; i<3 ; i++) grasp_specific_translation[i] = list->get(i).asDouble();
                for(int i=0 ; i<4 ; i++) grasp_specific_orientation[i] = list->get(3+i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid grasp_trsfm_right dimension in config. Should be 7.");
                valid_grasp_specific_transform = false;
            }
        }
        else valid_grasp_specific_transform = false;

        if( (!valid_grasp_specific_transform) && ((robot == "icubSim") || (robot == "icub")) )
        {
            yInfo() << "Loading grasp_trsfm_right default value for iCub";
            grasp_specific_translation[0] = -0.01;
            grasp_specific_orientation[1] = 1;
            grasp_specific_orientation[3] = - 38.0 * M_PI/180.0;
        }

        grasper_specific_transform_right = axis2dcm(grasp_specific_orientation);
        grasper_specific_transform_right.setSubcol(grasp_specific_translation, 0,3);
        yInfo() << "Grabber specific transform for right arm loaded\n" << grasper_specific_transform_right.toString();

        grasp_specific_translation.zero();
        grasp_specific_orientation.zero();
        list = rf.find("grasp_trsfm_left").asList();
        valid_grasp_specific_transform = true;

        if(list)
        {
            if(list->size() == 7)
            {
                for(int i=0 ; i<3 ; i++) grasp_specific_translation[i] = list->get(i).asDouble();
                for(int i=0 ; i<4 ; i++) grasp_specific_orientation[i] = list->get(3+i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid grasp_trsfm_left dimension in config. Should be 7.");
                valid_grasp_specific_transform = false;
            }
        }
        else valid_grasp_specific_transform = false;

        if( (!valid_grasp_specific_transform) && ((robot == "icubSim") || (robot == "icub")) )
        {
            yInfo() << "Loading grasp_trsfm_left default value for iCub";
            grasp_specific_translation[0] = -0.01;
            grasp_specific_orientation[1] = 1;
            grasp_specific_orientation[3] = + 38.0 * M_PI/180.0;
        }

        grasper_specific_transform_left = axis2dcm(grasp_specific_orientation);
        grasper_specific_transform_left.setSubcol(grasp_specific_translation, 0,3);
        yInfo() << "Grabber specific transform for left arm loaded\n" << grasper_specific_transform_left.toString();

        list = rf.find("approach_right").asList();
        if(list)
        {
            if(list->size() == 4)
            {
                for(int i=0 ; i<4 ; i++) grasper_approach_parameters_right[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid approach_right dimension in config. Should be 4.");
            }
        }
        else if((robot == "icubSim") || (robot == "icub"))
        {
            grasper_approach_parameters_right[0] = -0.05;
            grasper_approach_parameters_right[1] = 0.0;
            grasper_approach_parameters_right[2] = -0.05;
            grasper_approach_parameters_right[3] = 0.0;
        }
        yInfo() << "Grabber specific approach for right arm loaded\n" << grasper_approach_parameters_right.toString();

        list = rf.find("approach_left").asList();
        if(list)
        {
            if(list->size() == 4)
            {
                for(int i=0 ; i<4 ; i++) grasper_approach_parameters_left[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid approach_left dimension in config. Should be 4.");
            }
        }
        else if((robot == "icubSim") || (robot == "icub"))
        {
            grasper_approach_parameters_left[0] = -0.05;
            grasper_approach_parameters_left[1] = 0.0;
            grasper_approach_parameters_left[2] = +0.05;
            grasper_approach_parameters_left[3] = 0.0;
        }
        yInfo() << "Grabber specific approach for left arm loaded\n" << grasper_approach_parameters_left.toString();

        list = rf.find("grasp_bounding_box_right").asList();
        if(list)
        {
            if(list->size() == 6)
            {
                for(int i=0 ; i<6 ; i++) grasper_bounding_box_right[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid grasp_bounding_box_right dimension in config. Should be 6.");
            }
        }
        yInfo() << "Right grabber bounding box loaded\n" << grasper_bounding_box_right.toString();

        list = rf.find("grasp_bounding_box_left").asList();
        if(list)
        {
            if(list->size() == 6)
            {
                for(int i=0 ; i<6 ; i++) grasper_bounding_box_left[i] = list->get(i).asDouble();
            }
            else
            {
                yError() << prettyError(__FUNCTION__, "Invalid grasp_bounding_box_left dimension in config. Should be 6.");
            }
        }
        yInfo() << "Left grabber bounding box loaded\n" << grasper_bounding_box_left.toString();

        list = rf.find("planar_obstacle").asList();
        if(list)
        {
            if(list->size() == 4)
            {
                for(int i=0 ; i<4 ; i++) planar_obstacle[i] = list->get(i).asDouble();
            }
            else
            {
                planar_obstacle[0] = 0.0;
                planar_obstacle[1] = 0.0;
                planar_obstacle[2] = 1;
                planar_obstacle[3] = -(-0.15);
                yError() << prettyError(__FUNCTION__, "Invalid planar_obstacle dimension in config. Should be 4.");
            }
        }
        yInfo() << "Planar obstacle loaded\n" << planar_obstacle.toString();


        obstacle_safety_distance = rf.check("obstacle_safety_distance", Value(0.0)).asDouble();
        yInfo() << "Obstacle safety distance loaded=" << obstacle_safety_distance;

        position_error_threshold = rf.check("position_error_threshold", Value(0.01)).asDouble();
        yInfo() << "Position error threshold loaded=" << position_error_threshold;

        roundness_threshold = rf.check("roundness_threshold", Value(1.0)).asDouble();
        yInfo() << "Roundness threshold loaded=" << roundness_threshold;

        nb_cardinal_levels = rf.check("nb_cardinal_levels", Value(1)).asInt();
        if(nb_cardinal_levels < 1)
        {
            nb_cardinal_levels = 1;
        }
        yInfo() << "Number of cardinal point levels loaded=" << nb_cardinal_levels;

        //  open the necessary ports
        superq_rpc.open("/" + moduleName + "/superquadricRetrieve:rpc");
        point_cloud_rpc.open("/" + moduleName + "/pointCloud:rpc");
        action_render_rpc.open("/" + moduleName + "/actionRenderer:rpc");
        reach_calib_rpc.open("/" + moduleName + "/reachingCalibration:rpc");
        table_calib_rpc.open("/" + moduleName + "/tableCalib:rpc");
        module_rpc.open("/" + moduleName + "/cmd:rpc");

        //  open clients when using iCub

        if((robot == "icubSim") || (robot == "icub"))
        {
            Property optionLeftArm, optionRightArm;

            optionLeftArm.put("device", "cartesiancontrollerclient");
            optionLeftArm.put("remote", "/" + robot + "/cartesianController/left_arm");
            optionLeftArm.put("local", "/" + moduleName + "/cartesianClient/left_arm");

            optionRightArm.put("device", "cartesiancontrollerclient");
            optionRightArm.put("remote", "/" + robot + "/cartesianController/right_arm");
            optionRightArm.put("local", "/" + moduleName + "/cartesianClient/right_arm");

            if ((control_arms=="both") || (control_arms=="left"))
            {
                if (!left_arm_client.open(optionLeftArm))
                {
                    yError() << prettyError( __FUNCTION__, "Could not open cartesian solver client for left arm");
                    return false;
                }
            }
            if ((control_arms=="both") || (control_arms=="right"))
            {
                if (!right_arm_client.open(optionRightArm))
                {
                    if (left_arm_client.isValid())
                    {
                        left_arm_client.close();
                    }
                    yError() << prettyError( __FUNCTION__, "Could not open cartesian solver client for right arm");
                    return false;
                }
            }
        }

        //  attach callback
        attach(module_rpc);

        halt_requested = false;

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
        //vtk_renderer->SetBackground(0.1, 0.2, 0.2);
        vtk_renderer->SetBackground(0.8, 0.8, 0.8);

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
        table_calib_rpc.close();

        if (left_arm_client.isValid())
        {
            left_arm_client.close();
        }
        if (right_arm_client.isValid())
        {
            right_arm_client.close();
        }

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
        bool fixate_object = false;

        if (command.get(0).toString() == "grasp_pose")
        {
            //  normal operation
            if (command.size() == 3 || command.size() == 4)
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
                if (command.size() == 4 && command.get(3).toString() == "gaze")
                {
                    fixate_object = true;
                }

            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
            PointCloud<DataXYZRGBA> pc;
            yDebug() << "Requested object: " << obj;
            if (requestRefreshPointCloud(pc, obj, fixate_object))
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
            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            //  obtain grasp and render it
            if (command.size() == 3 || command.size() == 4)
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
                if (command.size() == 4 && command.get(3).toString() == "gaze")
                {
                    fixate_object = true;
                }
            }
            else
            {
                yError() << prettyError( __FUNCTION__,  "Invalid command size");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            PointCloud<DataXYZRGBA> pc;
            yDebug() << "Requested object: " << obj;

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if (!requestRefreshPointCloud(pc, obj, fixate_object))
            {
                yError() << prettyError( __FUNCTION__,  "Could not refresh point cloud.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if (!requestRefreshSuperquadric(pc))
            {
                yError() << prettyError( __FUNCTION__,  "Could not refresh superquadric.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if (!computeGraspPose(grasp_pose))
            {
                yError() << prettyError( __FUNCTION__,  "Could not compute grasping pose.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            yInfo() << "Pose retrieved: " << grasp_pose.toString();

            if (!executeGrasp(grasp_pose))
            {
                yError() << prettyError( __FUNCTION__,  "Could not perform the grasping.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            yInfo() << obj << "grasped";
            reply.addVocab(Vocab::encode("ack"));
            return true;
        }

        if (command.get(0).toString() == "grasp_from_position")
        {
            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            //  obtain grasp and render it
            Vector position3d(3, 0.0);
            if (command.size() == 3 || command.size() == 4)
            {
                Bottle *position3d_bottle = command.get(1).asList();

                if(!position3d_bottle)
                {
                    yError() << prettyError( __FUNCTION__,  "Invalid position vector format. Should be a list of double.");
                    reply.addVocab(Vocab::encode("nack"));
                    return true;
                }

                if(position3d_bottle->size() != 3)
                {
                    yError() << prettyError( __FUNCTION__,  "Invalid position vector size. Should be 3.");
                    reply.addVocab(Vocab::encode("nack"));
                    return true;
                }

                for(int i=0 ; i<3 ; i++) position3d[i] = position3d_bottle->get(i).asDouble();

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
                if (command.size() == 4 && command.get(3).toString() == "gaze")
                {
                    fixate_object = true;
                }
            }
            else
            {
                yError() << prettyError( __FUNCTION__,  "Invalid command size.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            PointCloud<DataXYZRGBA> pc;
            if (!requestRefreshPointCloudFromPosition(pc, position3d, fixate_object))
            {
                yError() << prettyError( __FUNCTION__,  "Could not refresh point cloud.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if (!requestRefreshSuperquadric(pc))
            {
                yError() << prettyError( __FUNCTION__,  "Could not refresh superquadric.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if (!computeGraspPose(grasp_pose))
            {
                yError() << prettyError( __FUNCTION__,  "Could not compute grasping pose.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            if(halt_requested)
            {
                yInfo() << "Halt requested before end of process";
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            yInfo() << "Pose retrieved: " << grasp_pose.toString();

            if (!executeGrasp(grasp_pose))
            {
                yError() << prettyError( __FUNCTION__,  "Could not perform the grasping.");
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }

            yInfo() << "Object grasped";
            reply.addVocab(Vocab::encode("ack"));
            return true;
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
                yError() << prettyError( __FUNCTION__,  "Unable to open file");
                reply.addVocab(Vocab::encode("nack"));
                return false;
            }

            //  parse the OFF file line by line
            string line;
            getline(file, line);
            if (line != "COFF")
            {
                yError() << prettyError( __FUNCTION__,  "File parsing failed");
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

        if (command.get(0).toString() == "drop")
        {
            //  Just relay the command forward to actionsRenderingEngine
            if (action_render_rpc.getOutputCount() > 0)
            {
                Bottle cmd, reply;
                cmd.addVocab(Vocab::encode("drop"));
                action_render_rpc.write(cmd, reply);
                cmd_success = (reply.get(0).asVocab() == Vocab::encode("ack"));
            }
            else
            {
                cmd_success = false;
            }
        }

        if (command.get(0).toString() == "home")
        {
            //  Just relay the command forward to actionsRenderingEngine
            if (action_render_rpc.getOutputCount() > 0)
            {
                Bottle cmd, reply;
                cmd.addVocab(Vocab::encode("home"));
                action_render_rpc.write(cmd, reply);
                cmd_success = (reply.get(0).asVocab() == Vocab::encode("ack"));
            }
            else
            {
                cmd_success = false;
            }
        }

        if (command.get(0).toString() == "get_raw_grasp_poses")
        {
            // compute raw grasp poses candidates from superquadric parameters
            // superquadric parameters (center_x center_y center_z rot_axis_x rot_axis_y rot_axis_z angle axis_size_1 axis_size_2 axis_size_3 roundness_1 roundness_2)
            // hand to use "right"/"left"
            if (command.size() > 12)
            {
                if (command.size() > 13)
                {
                    hand = command.get(13).toString();
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

                Vector super_quadric_parameters(12);
                for(int i=0 ; i<12 ; i++) super_quadric_parameters[i] = command.get(i+1).asDouble();

                vector<Matrix> raw_grasp_pose_candidates;
                this->computeRawGraspPoseCandidates(super_quadric_parameters, raw_grasp_pose_candidates);

                for(int i=0 ; i<raw_grasp_pose_candidates.size() ; i++)
                {
                    for(int j=0 ; j<3 ; j++) reply.addDouble(raw_grasp_pose_candidates[i](j,3));
                    Vector orientation = dcm2axis(raw_grasp_pose_candidates[i]);
                    for(int j=0 ; j<4 ; j++) reply.addDouble(orientation[j]);
                }
                return true;
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
        }

        if (command.get(0).toString() == "refine_grasp_pose")
        {
            // refine a grasp pose candidate to be compatible with the robot constraints
            // superquadric parameters (center_x center_y center_z rot_axis_x rot_axis_y rot_axis_z angle axis_size_1 axis_size_2 axis_size_3)
            // pose candidate (t_x t_y t_z rot_axis_x rot_axis_y rot_axis_z angle)
            // hand to use "right"/"left"
            if (command.size() > 17)
            {
                if (command.size() > 18)
                {
                    hand = command.get(18).toString();
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

                Vector super_quadric_parameters(10);
                for(int i=0 ; i<10 ; i++) super_quadric_parameters[i] = command.get(i+1).asDouble();

                Vector orientation(4, 0.0);
                for(int i=0 ; i<4 ; i++) orientation[i] = command.get(i+14).asDouble();

                Matrix raw_grasp_pose_candidate = axis2dcm(orientation);
                raw_grasp_pose_candidate(3,3) = 1;
                for(int i=0 ; i<3 ; i++) raw_grasp_pose_candidate(i,3) = command.get(i+11).asDouble();

                vector<Matrix> raw_grasp_pose_candidates;
                raw_grasp_pose_candidates.push_back(raw_grasp_pose_candidate);

                vector<Matrix> refined_grasp_pose_candidates;
                this->refineGraspPoseCandidates(super_quadric_parameters, raw_grasp_pose_candidates, refined_grasp_pose_candidates);

                if((refined_grasp_pose_candidates.front().cols()==4) && (refined_grasp_pose_candidates.front().rows()==4))
                {
                    reply.addVocab(Vocab::encode("ok"));
                    for(int i=0 ; i<3 ; i++) reply.addDouble(refined_grasp_pose_candidates.front()(i,3));
                    Vector orientation = dcm2axis(refined_grasp_pose_candidates.front());
                    for(int i=0 ; i<4 ; i++) reply.addDouble(orientation[i]);
                }
                else
                {
                    reply.addVocab(Vocab::encode("nok"));
                }
                return true;
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
        }

        if (command.get(0).toString() == "select_best_grasp_pose")
        {
            // select the best grasp pose amoung a list of candidate in order to be optimize wit respect to the robot kinematics and object shape
            // superquadric parameters (center_x center_y center_z rot_axis_x rot_axis_y rot_axis_z angle axis_size_1 axis_size_2 axis_size_3)
            // list of pose candidates (t_x t_y t_z rot_axis_x rot_axis_y rot_axis_z angle)
            // hand to use "right"/"left"

            if (command.size() > 17)
            {
                Vector super_quadric_parameters(10);
                for(int i=0 ; i<10 ; i++) super_quadric_parameters[i] = command.get(i+1).asDouble();

                vector<Matrix> grasp_pose_candidates;
                for(int i=11 ; i+6<command.size() ; i+=7)
                {
                    Vector orientation(4, 0.0);
                    for(int j=0 ; j<4 ; j++) orientation[j] = command.get(i+3+j).asDouble();

                    Matrix raw_grasp_pose_candidate = axis2dcm(orientation);

                    for(int j=0 ; j<3 ; j++) raw_grasp_pose_candidate(j,3) = command.get(i+j).asDouble();

                    grasp_pose_candidates.push_back(raw_grasp_pose_candidate);
                }

                if (command.size() > 10 + 7*grasp_pose_candidates.size() + 1)
                {
                    hand = command.get(10 + 7*grasp_pose_candidates.size() +1).asString();
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

                int best_grasp_pose_index;
                vector<Vector> costs;
                if(this->getBestCandidatePose(super_quadric_parameters, grasp_pose_candidates, best_grasp_pose_index, costs))
                {
                    Vector best_pose(7, 0.0);
                    best_pose.setSubvector(0, grasp_pose_candidates[best_grasp_pose_index].subcol(0,3,3));
                    best_pose.setSubvector(3, yarp::math::dcm2axis(grasp_pose_candidates[best_grasp_pose_index].submatrix(0,2, 0,2)));

                    reply.addVocab(Vocab::encode("ok"));
                    for(int j=0 ; j<best_pose.size() ; j++) reply.addDouble(best_pose[j]);
                }
                else
                {
                    reply.addVocab(Vocab::encode("nok"));
                }
                return true;
            }
            else
            {
                reply.addVocab(Vocab::encode("nack"));
                return true;
            }
        }

        if (command.get(0).toString() == "restart")
        {
            halt_requested = false;
            reply.addVocab(Vocab::encode("ack"));
            return true;
        }

        if (command.get(0).toString() == "halt")
        {
            halt_requested = true;
            reply.addVocab(Vocab::encode("ack"));
            return true;
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

           double bb = 0.0;
           for (size_t i=0; i<centroid.size(); i++)
           {
               centroid[i] = 0.5 * (bounds[i<<1] + bounds[(i<<1)+1]);
               bb = std::max(bb, bounds[(i<<1)+1] - bounds[i<<1]);
           }
           bb *= 3.0;

           vtk_camera->SetPosition(centroid[0] + bb, centroid[1], centroid[2] + bb);
           vtk_camera->SetViewUp(0.0, 0.0, 1.0);
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
    bool requestRefreshPointCloud(PointCloud<DataXYZRGBA> &point_cloud, const string &object, const bool &fixate_object = false)
    {
        //  query point-cloud-read via rpc for the point cloud
        //  command: get_point_cloud objectName
        //  put point cloud into container, return true if operation was ok
        //  or call refreshpointcloud
        Bottle cmd_request;
        Bottle cmd_reply;

        //  if fixate_object is given, look at the object before acquiring the point cloud
        if (fixate_object)
        {
            if(action_render_rpc.getOutputCount()<1)
            {
                yError() << prettyError( __FUNCTION__,  "No connection to action rendering module");
                return false;
            }

            cmd_request.addVocab(Vocab::encode("look"));
            cmd_request.addString(object);
            cmd_request.addString("wait");

            action_render_rpc.write(cmd_request, cmd_reply);
            if (cmd_reply.get(0).asVocab() != Vocab::encode("ack"))
            {
                yError() << prettyError( __FUNCTION__,  "Didn't manage to look at the object");
                return false;
            }
        }

        point_cloud.clear();
        cmd_request.clear();
        cmd_reply.clear();

        cmd_request.addString("get_point_cloud");
        cmd_request.addString(object);

        if(point_cloud_rpc.getOutputCount()<1)
        {
            yError() << prettyError( __FUNCTION__,  "No connection to point cloud module");
            return false;
        }

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
            yError() << prettyError( __FUNCTION__,  "Point cloud null or empty");
            return false;
        }

    }

    /****************************************************************/
    bool requestRefreshPointCloudFromPosition(PointCloud<DataXYZRGBA> &point_cloud, const Vector &position, const bool &fixate_object = false)
    {
        //  query point-cloud-read via rpc for the point cloud
        //  command: get_point_cloud_from_3D_position pos_x pos_y pos_z
        //  put point cloud into container, return true if operation was ok
        //  or call refreshpointcloud

        if(position.size() < 3)
        {
            yError() << prettyError( __FUNCTION__,  "Invalid position vector dimension. Should be 3.");
            return false;
        }

        Bottle cmd_request;
        Bottle cmd_reply;

        //  if fixate_object is given, look at the object before acquiring the point cloud

        if (fixate_object)
        {
            if(action_render_rpc.getOutputCount()<1)
            {
                yError() << prettyError( __FUNCTION__,  "No connection to action rendering module");
                return false;
            }

            cmd_request.addVocab(Vocab::encode("look"));
            Bottle &subcmd_request = cmd_request.addList();
            subcmd_request.addString("cartesian");
            for(int i=0 ; i<3 ; i++) subcmd_request.addDouble(position[i]);
            cmd_request.addString("wait");

            action_render_rpc.write(cmd_request, cmd_reply);
            if (cmd_reply.get(0).asVocab() != Vocab::encode("ack"))
            {
                yError() << prettyError( __FUNCTION__,  "Didn't manage to look at the object");
                return false;
            }
        }

        point_cloud.clear();
        cmd_request.clear();
        cmd_reply.clear();

        cmd_request.addString("get_point_cloud_from_3D_position");
        for(int i=0 ; i<3 ; i++) cmd_request.addDouble(position[i]);

        if(point_cloud_rpc.getOutputCount()<1)
        {
            yError() << prettyError( __FUNCTION__,  "No connection to point cloud module");
            return false;
        }

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
            yError() << prettyError( __FUNCTION__,  "Point cloud null or empty");
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

        if(superq_rpc.getOutputCount()<1)
        {
            yError() << prettyError( __FUNCTION__,  "requestRefreshSuperquadric: no connection to superquadric module");
            return false;
        }

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
            yError() << prettyError( __FUNCTION__,  "Retrieved superquadric is invalid!") << superq_parameters.toString();
            return false;
        }

    }

    /****************************************************************/
    bool isCandidateGraspFeasible(const Vector &super_quadric_parameters, const Matrix &candidate_pose)
    {
        //  filter candidate grasp. True for good grasp

        if(super_quadric_parameters.size() < 10)
        {
            yError() << prettyError( __FUNCTION__,  "isCandidateGraspFeasible: invalid superquadric parameters vector dimensions");
            return false ;
        }

        Vector root_z_axis(3, 0.0);
        root_z_axis(2) = 1;

        Vector superq_XYZW_orientation = super_quadric_parameters.subVector(3,6);
        Matrix superq_mat_orientation = axis2dcm(superq_XYZW_orientation).submatrix(0,2, 0,2);
        Vector superq_axes_size = super_quadric_parameters.subVector(7,9);
        Matrix pose_mat_rotation = candidate_pose.submatrix(0,2, 0,2);

        Matrix pose_superq_mat_rotation = pose_mat_rotation.transposed() * superq_mat_orientation;
        Vector pose_ax_size(3, 0.0);
        for(int i=0 ; i<3 ; i++)
        {
            for(int j=0 ; j<3 ; j++)
            {
                double v = pose_superq_mat_rotation(i,j) * superq_axes_size[j];
                pose_ax_size[i] += v*v;
            }
            pose_ax_size[i] = sqrt(pose_ax_size[i]);
        }

        /*
         * Filtering parameters:
         * 1 - object large enough for grasping
         * 2 - object small enough for grasping
         * 3 - thumb cannot point down
         * 4 - palm cannot point up
         */

        bool ok1=true, ok2=true, ok3=false, ok4=false;
        for(int i=0 ; i<3 ; i++)
        {
            double object_size = 2*pose_ax_size[i];
            ok1 &= (object_size > min_object_size[i]);
            ok2 &= (object_size < max_object_size[i]);
        }

        ok3 = dot(pose_mat_rotation.getCol(1), root_z_axis) <= 0.1;
        if (grasping_hand == WhichHand::HAND_RIGHT)
        {
            //  ok if hand z axis points downward
            ok4 = (dot(pose_mat_rotation.getCol(2), root_z_axis) <= 0.1);
        }
        else
        {
            //  ok if hand z axis points upwards
            ok4 = (dot(pose_mat_rotation.getCol(2), root_z_axis) >= -0.1);
        }

        return (ok1 && ok2 && ok3 && ok4);
    }

    /****************************************************************/
    bool getPoseCostFunction(const Vector &super_quadric_parameters, const Matrix &candidate_pose, Vector &cost)
    {
        cost.resize(2, std::numeric_limits<double>::max());

        if(super_quadric_parameters.size() < 10)
        {
            yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: invalid superquadric parameters vector dimensions");
            return false ;
        }

        //  compute precision for movement

        Matrix pose_mat_rotation = candidate_pose.submatrix(0,2, 0,2);
        Vector x_d = candidate_pose.subcol(0,3,3);
        Vector o_d = dcm2axis(pose_mat_rotation);
        Vector x_d_hat, o_d_hat, q_d_hat;

        if((robot == "icubSim") || (robot == "icub"))
        {
            if ((grasping_hand == WhichHand::HAND_LEFT) && left_arm_client.isValid())
            {
                left_arm_client.view(icart);
            }
            else if ((grasping_hand == WhichHand::HAND_RIGHT) && right_arm_client.isValid())
            {
                right_arm_client.view(icart);
            }
            else
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: Invalid arm selected for kinematic!");
                return false;
            }

            //  store the context for the previous iKinCartesianController config
            int context_backup;
            icart->storeContext(&context_backup);

            //  set up the context for the computation of the candidates
            this->setGraspContext();

            bool success = icart->askForPose(x_d, o_d, x_d_hat, o_d_hat, q_d_hat);

            //  restore previous context
            icart->restoreContext(context_backup);
            icart->deleteContext(context_backup);

            if(!success)
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: could not communicate with kinematics module");
                return false;
            }
        }
        else
        {
            if(action_render_rpc.getOutputCount()<1)
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: no connection to action rendering module");
                return false;
            }

            Bottle cmd, reply;
            cmd.addVocab(Vocab::encode("ask"));
            Bottle &subcmd = cmd.addList();
            for(int i=0 ; i<3 ; i++) subcmd.addDouble(x_d[i]);
            for(int i=0 ; i<4 ; i++) subcmd.addDouble(o_d[i]);
            if(grasping_hand == WhichHand::HAND_LEFT)
            {
                cmd.addString("left");
            }
            else if(grasping_hand == WhichHand::HAND_RIGHT)
            {
                cmd.addString("right");
            }
            action_render_rpc.write(cmd, reply);

            if(reply.size()<1)
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: empty reply from action rendering module");
                return false;
            }

            if(reply.get(0).asVocab() != Vocab::encode("ack"))
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: invalid reply from action rendering module:") << reply.toString();
                return false;
            }

            if(reply.size()<3)
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: invlaid reply size from action rendering module") << reply.toString();
                return false;
            }

            if(!reply.check("q"))
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: invalid reply from action rendering module: missing q:") << reply.toString();
                return false;
            }

            Bottle *joints = reply.find("q").asList();
            q_d_hat.resize(joints->size());
            for(int i=0 ; i<joints->size() ; i++) q_d_hat[i] = joints->get(i).asDouble();

            if(!reply.check("x"))
            {
                yError() << prettyError( __FUNCTION__,  "getPoseCostFunction: invalid reply from action rendering module: missing x:") << reply.toString();
                return false;
            }

            Bottle *position = reply.find("x").asList();
            x_d_hat.resize(3);
            for(int i=0 ; i<3 ; i++) x_d_hat[i] = position->get(i).asDouble();
            o_d_hat.resize(4);
            for(int i=0 ; i<4 ; i++) o_d_hat[i] = position->get(3+i).asDouble();
        }

        yDebug() << "Requested: " << candidate_pose.toString();

        //  calculate position cost function (first component of cost function)
        cost[0] = norm(x_d - x_d_hat);

        //  calculate orientation cost function
        Matrix tmp = axis2dcm(o_d_hat).submatrix(0,2, 0,2);
        Matrix orientation_error_matrix =  pose_mat_rotation * tmp.transposed();
        Vector orientation_error_vector = dcm2axis(orientation_error_matrix);

        Vector superq_XYZW_orientation = super_quadric_parameters.subVector(3,6);
        Matrix superq_mat_orientation = axis2dcm(superq_XYZW_orientation).submatrix(0,2, 0,2);
        Vector superq_axes_size = super_quadric_parameters.subVector(7,9);

        Matrix pose_superq_mat_rotation = pose_mat_rotation.transposed() * superq_mat_orientation;
        Vector pose_ax_size(3, 0.0);
        for(int i=0 ; i<3 ; i++)
        {
            for(int j=0 ; j<3 ; j++)
            {
                double v = pose_superq_mat_rotation(i,j) * superq_axes_size[j];
                pose_ax_size[i] += v*v;
            }
            pose_ax_size[i] = sqrt(pose_ax_size[i]);
        }

        cost[1] = norm(orientation_error_vector.subVector(0,2)) * fabs(sin(orientation_error_vector(3)));

        cost[1] = 0.5*cost[1] + 0.5*(1-pose_ax_size[1]/yarp::math::findMax(pose_ax_size));

        yDebug() << "Cost function: " << cost.toString();

        return true;

    }

    /****************************************************************/
    void computeRawGraspPoseCandidates(const Vector &super_quadric_parameters, vector<Matrix> &raw_grasp_pose_candidates)
    {
        // compute grasping pose on the different side of the superquadric

        raw_grasp_pose_candidates.clear();

        if (super_quadric_parameters.size() < 12)
        {
            yError() << prettyError( __FUNCTION__,  "computeRawGraspPoseCandidates: invalid superquadric parameters vector dimensions");
            return;
        }

        Vector superq_center = super_quadric_parameters.subVector(0,2);
        Vector superq_XYZW_orientation = super_quadric_parameters.subVector(3,6);
        Vector superq_axes_size = super_quadric_parameters.subVector(7,9);
        Vector superq_roundness = super_quadric_parameters.subVector(10,11);

        //  get orientation of the superq in 3x3 rotation matrix form

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
                    Matrix candidate_pose(4,4);
                    candidate_pose(3,3) = 1;
                    //  create gz with cross product
                    //  create candidate entry
                    Vector gz = cross(gx, gy);

                    candidate_pose.setSubcol(gx, 0,0);
                    candidate_pose.setSubcol(gy, 0,1);
                    candidate_pose.setSubcol(gz, 0,2);
                    double s = (grasping_hand == WhichHand::HAND_RIGHT ? -1.0 : 1.0);
                    candidate_pose.setSubcol(superq_center + s * superq_axes_size(3 - idx/2 - jdx/2) / norm(gz) * gz, 0,3);

                    raw_grasp_pose_candidates.push_back(candidate_pose);
                }
            }
        }
    }

    /****************************************************************/
    void refineGraspPoseCandidates(const Vector &super_quadric_parameters, const vector<Matrix> &raw_grasp_pose_candidates, vector<Matrix> &refined_grasp_pose_candidates)
    {
        // modify the grasping poses to account for the robot and table  geometry and prune the poses that are not realistic

        refined_grasp_pose_candidates.clear();

        if(super_quadric_parameters.size() < 10)
        {
            yError() << prettyError( __FUNCTION__,  "refineGraspPoseCandidates: invalid superquadric parameters vector dimensions");
            return;
        }

        Vector grasper_bounding_box;
        if(grasping_hand == WhichHand::HAND_RIGHT)
        {
            grasper_bounding_box = grasper_bounding_box_right;
        }
        else
        {
            grasper_bounding_box = grasper_bounding_box_left;
        }

        int cnt = 0;
        for(size_t idx=0 ; idx<raw_grasp_pose_candidates.size() ; idx++)
        {
            Matrix pose_candidate = raw_grasp_pose_candidates[idx];

            // check if pose is feasible with respect to grasper/object size and some grasper orientation constraints
            if (isCandidateGraspFeasible(super_quadric_parameters, pose_candidate))
            {
                // apply robot specific transform

                if(grasping_hand == WhichHand::HAND_RIGHT)
                {
                    pose_candidate = pose_candidate * grasper_specific_transform_right;
                }
                else
                {
                    pose_candidate = pose_candidate * grasper_specific_transform_left;
                }

                // check collision with plane

                double min_distance_to_plane = 0.0;

                for(int i=0 ; i<2 ; i++)
                {
                    Vector corner(4, 1.0);
                    corner[0] = grasper_bounding_box[i];
                    for(int j=0 ; j<2 ; j++)
                    {
                        corner[1] = grasper_bounding_box[2+j];
                        for(int k=0 ; k<2 ; k++)
                        {
                            corner[2] = grasper_bounding_box[4+k];
                            Vector corner_world = pose_candidate * corner;

                            double distance = dot(corner_world, planar_obstacle);
                            if(distance < min_distance_to_plane)
                            {
                                min_distance_to_plane = distance;
                            }
                        }
                    }
                }

                //  if the grasp is close to the surface, we need to adjust the pose to avoid collision
                if(min_distance_to_plane < 0.0)
                {
                    for(int i=0 ; i<3 ; i++)
                    {
                        pose_candidate(i,3) += (-min_distance_to_plane+obstacle_safety_distance) * planar_obstacle[i];
                    }
                }

                refined_grasp_pose_candidates.push_back(pose_candidate);
                cnt++;
            }
            else
            {
                refined_grasp_pose_candidates.push_back(Matrix());
            }
        }

        yInfo() <<  "getGraspingPoseCandidates: keep" << cnt << "/" << raw_grasp_pose_candidates.size() << "feasible grasping pose candidates";
    }

    /****************************************************************/
    void computeGraspCandidates(const Vector &superq_parameters, vector<Matrix> &grasp_pose_candidates)
    {
        //  compute a series of viable grasp candidates according to superquadric parameters
        LockGuard lg(mutex);

        //  retrieve table height
        //  otherwise, leave default value
        bool table_ok = false;
        if (robot != "icubSim" && table_calib_rpc.getOutputCount() > 0)
        {
            Bottle table_cmd, table_rply;
            table_cmd.addVocab(Vocab::encode("get"));
            table_cmd.addString("table");

            table_calib_rpc.write(table_cmd, table_rply);
            if (Bottle *payload = table_rply.get(0).asList())
            {
                if (payload->size() >= 2)
                {
                    planar_obstacle[0] = 0.0;
                    planar_obstacle[1] = 0.0;
                    planar_obstacle[2] = 1.0;

                    planar_obstacle[3] = - payload->get(1).asDouble();
                    table_ok = true;
                }
            }
        }
        if (!table_ok)
        {
            yWarning() << "Unable to retrieve table height, using default.";
        }
        yInfo() << "Using table height =" << -planar_obstacle[3];

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

        vector<Matrix> raw_grasp_pose_candidates;
        this->computeRawGraspPoseCandidates(superq_parameters, raw_grasp_pose_candidates);

        this->refineGraspPoseCandidates(superq_parameters, raw_grasp_pose_candidates, grasp_pose_candidates);

        Vector superq_XYZW_orientation = superq_parameters.subVector(3,6);
        Vector superq_axes_size = superq_parameters.subVector(7,9);

        pose_candidates.clear();

        for(size_t idx=0 ; idx<grasp_pose_candidates.size() ; idx++)
        {
            shared_ptr<GraspPose> candidate_pose = shared_ptr<GraspPose>(new GraspPose);

            if((grasp_pose_candidates[idx].rows()!=4) || (grasp_pose_candidates[idx].cols()!=4))
            {
                pose_candidates.push_back(candidate_pose);
                continue;
            }

            candidate_pose->pose_translation = grasp_pose_candidates[idx].subcol(0,3,3);
            candidate_pose->pose_rotation = grasp_pose_candidates[idx].submatrix(0,2,0,2);

            Matrix superq_mat_orientation = axis2dcm(superq_XYZW_orientation).submatrix(0,2, 0,2);

            Matrix pose_superq_mat_rotation = raw_grasp_pose_candidates[idx].submatrix(0,2,0,2).transposed() * superq_mat_orientation;
            candidate_pose->pose_ax_size.zero();

            for(int i=0 ; i<3 ; i++)
            {
                for(int j=0 ; j<3 ; j++)
                {
                    double v = pose_superq_mat_rotation(i,j) * superq_axes_size[j];
                    candidate_pose->pose_ax_size[i] += v*v;
                }
                candidate_pose->pose_ax_size[i] = sqrt(candidate_pose->pose_ax_size[i]);
            }

            if (!candidate_pose->setHomogeneousTransform(candidate_pose->pose_rotation, candidate_pose->pose_translation))
            {
                yError() << prettyError( __FUNCTION__,  "Error setting homogeneous transform!");
                continue;
            }

            //  if candidate is good, set vtk transform and actor
            candidate_pose->setvtkTransform(candidate_pose->pose_transform);
            candidate_pose->pose_vtk_actor->SetUserTransform(candidate_pose->pose_vtk_transform);
            pose_actors[idx]->SetUserTransform(candidate_pose->pose_vtk_transform);

            //  fix graphical properties
//                        candidate_pose->pose_vtk_actor->AxisLabelsOff();
//                        candidate_pose->pose_vtk_actor->SetTotalLength(0.02, 0.02, 0.02);

            candidate_pose->pose_vtk_actor->ShallowCopy(pose_actors[idx]);
            pose_actors[idx]->AxisLabelsOff();
            pose_actors[idx]->SetTotalLength(0.02, 0.02, 0.02);
            pose_actors[idx]->VisibilityOn();

            //  add actor to renderer
            //vtk_renderer->AddActor(candidate_pose->pose_vtk_actor);
            //vtk_renderer->AddActor(candidate_pose->pose_vtk_caption_actor);
            pose_candidates.push_back(candidate_pose);
        }

        return;

    }

    /****************************************************************/
    bool getBestCandidatePose(const Vector &super_quadric_parameters, const vector<Matrix> &grasp_pose_candidates, int &best_pose_index, vector<Vector> &costs)
    {
        if (super_quadric_parameters.size() < 10)
        {
            yError() << prettyError( __FUNCTION__,  "getBestCandidatePose: invalid superquadric parameters vector dimensions");
            return false;
        }

        // compute costs

        costs.resize(grasp_pose_candidates.size(), Vector(2, std::numeric_limits<double>::max()));
        for(int i=0 ; i<grasp_pose_candidates.size() ; i++)
        {
            if((grasp_pose_candidates[i].rows()==4) || (grasp_pose_candidates[i].cols()==4))
            {
                if(!(this->getPoseCostFunction(super_quadric_parameters, grasp_pose_candidates[i], costs[i])))
                {
                    yError() << prettyError( __FUNCTION__,  "getBestCandidatePose: could not compute grasping pose cost function");
                    return false;
                }
            }
        }

        struct CostEntry
        {
            //  define a cost entry structure to better handle
            //  sorting of the costs
            double cost_position;
            double cost_orientation;
            int pose_original_index;
            Matrix pose;

            CostEntry(const Vector &cost, const Matrix &grasp_candidate, const int &idx): cost_position(cost(0)), cost_orientation(cost(1)), pose(grasp_candidate), pose_original_index(idx) {}

            bool operator < (const CostEntry& ent) const
            {
                //  the cost functions are sorted according to the orientation component
                return (cost_orientation < ent.cost_orientation);
            }

            bool operator > (const CostEntry& ent) const
            {
                return (cost_orientation > ent.cost_orientation);
            }
        };

        //  compose a vector of cost functions
        vector <CostEntry> sorted_costs;
        for (size_t i=0; i<costs.size(); i++)
        {
            if (costs[i][0] < position_error_threshold)
                sorted_costs.push_back(CostEntry(costs[i], grasp_pose_candidates[i], i));
        }

        //  sort them
        if (sorted_costs.size() > 0)
        {
            std::sort(sorted_costs.begin(), sorted_costs.end());

            //  best grasp pose is the first in the vector
            best_pose_index = sorted_costs[0].pose_original_index;
            yInfo() << "getBestCandidatePose: best candidate cost: " << costs[best_pose_index].toString();

            return true;
        }
        else
        {
            yWarning() << "getBestCandidatePose: no candidate pose is within position accuracy requirements.";

            return false;
        }

    }

    /****************************************************************/
    bool fixReachingOffset(const Vector &poseToFix, Vector &poseFixed,
                           const bool invert=false)
    {
        //  fix the pose offset accordint to iolReachingCalibration
        //  pose is supposed to be (x y z gx gy gz theta)
        if ((robot == "r1" || robot == "icub") && reach_calib_rpc.getOutputCount() > 0)
        {
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
            command.addInt(invert?1:0);         //  flag to invert input/output map

            reach_calib_rpc.write(command, reply);

            //  incoming reply is going to be (success x y z)
            if (reply.size() < 2)
            {
                yError() << prettyError( __FUNCTION__,  "Failure retrieving fixed pose");
                return false;
            }
            else if (reply.get(0).asVocab() == Vocab::encode("ok"))
            {
                poseFixed = poseToFix;
                poseFixed(0) = reply.get(1).asDouble();
                poseFixed(1) = reply.get(2).asDouble();
                poseFixed(2) = reply.get(3).asDouble();
                return true;
            }
            else
            {
                yWarning() << "Couldn't retrieve fixed pose. Continuing with unchanged pose";
            }
        }
        else
        {
            //  if we are working with the simulator or there is no calib map, the pose doesn't need to be corrected
            poseFixed = poseToFix;
            yWarning() << "Connection to iolReachingCalibration not detected or calibration map not present: pose will not be changed";
            return true;
        }
    }

    /****************************************************************/
    bool computeGraspPose(Vector &pose)
    {
        //  execute the pose computation pipeline
        bool success = false;

        //  if anything goes wrong, the operation fails and the function
        //  returns a failure

        //  get superquadric parameters
        Vector superq_parameters(12);
        Vector superq_center = vtk_superquadric->getCenter();
        superq_parameters.setSubvector(0, superq_center);
        Vector superq_XYZW_orientation = vtk_superquadric->getOrientationXYZW();
        superq_XYZW_orientation[3] *= (M_PI / 180.0);
        superq_parameters.setSubvector(3, superq_XYZW_orientation);
        Vector superq_axes_size = vtk_superquadric->getAxesSize();
        superq_parameters.setSubvector(7, superq_axes_size);
        Vector superq_roundness = vtk_superquadric->getRoundness();
        superq_parameters.setSubvector(10, superq_roundness);

        vector<Matrix> grasp_pose_candidates;
        computeGraspCandidates(superq_parameters, grasp_pose_candidates);

        int best_grasp_pose_index;
        vector<Vector> costs;
        if(getBestCandidatePose(superq_parameters, grasp_pose_candidates, best_grasp_pose_index, costs))
        {
            Vector best_pose(7, 0.0);
            best_pose.setSubvector(0, grasp_pose_candidates[best_grasp_pose_index].subcol(0,3,3));
            best_pose.setSubvector(3, yarp::math::dcm2axis(grasp_pose_candidates[best_grasp_pose_index].submatrix(0,2, 0,2)));

            yInfo() << ": cartesian " << best_pose.subVector(0,2).toString()
                    << " pose " << best_pose.subVector(3,6).toString();

            for(size_t idx=0 ; (idx<pose_candidates.size() && idx<grasp_pose_candidates.size()) ; idx++)
            {
                if((grasp_pose_candidates[idx].rows()!=4) || (grasp_pose_candidates[idx].cols()!=4)) continue;

                pose_candidates[idx]->pose_cost_function = costs[idx];
                stringstream ss;
                ss << fixed << setprecision(3) << pose_candidates[idx]->pose_cost_function(0) << "_" << fixed << setprecision(3) << pose_candidates[idx]->pose_cost_function(1);
                pose_candidates[idx]->setvtkActorCaption(ss.str());

                pose_captions[idx]->VisibilityOn();
                pose_captions[idx]->GetTextActor()->SetTextScaleModeToNone();
                pose_captions[idx]->SetCaption(pose_candidates[idx]->pose_vtk_caption_actor->GetCaption());
                pose_captions[idx]->BorderOff();
                pose_captions[idx]->LeaderOn();
                pose_captions[idx]->GetCaptionTextProperty()->SetFontSize(15);
                pose_captions[idx]->GetCaptionTextProperty()->FrameOff();
                pose_captions[idx]->GetCaptionTextProperty()->ShadowOff();
                pose_captions[idx]->GetCaptionTextProperty()->BoldOff();
                pose_captions[idx]->GetCaptionTextProperty()->ItalicOff();
                pose_captions[idx]->GetCaptionTextProperty()->SetColor(0.1, 0.1, 0.1);
                pose_captions[idx]->SetAttachmentPoint(pose_candidates[idx]->pose_vtk_caption_actor->GetAttachmentPoint());

                if(idx==best_grasp_pose_index)
                {
                    pose_captions[idx]->GetCaptionTextProperty()->SetColor(0.0, 1.0, 0.0);
                    pose_captions[idx]->GetCaptionTextProperty()->BoldOn();
                }
            }

            success = fixReachingOffset(best_pose, pose);
        }

        return success;
    }

    /****************************************************************/
    bool executeGrasp(Vector &pose)
    {
        if(robot == "icubSim")
        {
            //  simulation context, suppose there is no actionsRenderingEngine running
            int context_backup;
            icart->storeContext(&context_backup);
            setGraspContext();
            Vector previous_x(3), previous_o(4);
            icart->getPose(previous_x, previous_o);
            icart->goToPoseSync(pose.subVector(0, 2), pose.subVector(3,6));
            icart->waitMotionDone();
            icart->goToPoseSync(previous_x, previous_o);
            icart->waitMotionDone();
            icart->restoreContext(context_backup);
            icart->deleteContext(context_backup);
            return true;
        }
        else
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
            if (grasping_hand == WhichHand::HAND_LEFT)
            {
                for(int i=0 ; i<4 ; i++) ptr2.addDouble(grasper_approach_parameters_left[i]);
                command.addString("left");
            }
            else
            {
                for(int i=0 ; i<4 ; i++) ptr2.addDouble(grasper_approach_parameters_right[i]);
                command.addString("right");
            }

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

    }


public:
    GraspProcessorModule(): closing(false), halt_requested(false),
        planar_obstacle(4, 0.0), grasper_bounding_box_right(6, 0.0), grasper_bounding_box_left(6, 0.0), obstacle_safety_distance(0.005),
        grasping_hand(WhichHand::HAND_RIGHT), min_object_size(3, 0.0), max_object_size(3, std::numeric_limits<double>::max()),
        grasper_specific_transform_right(eye(4,4)), grasper_specific_transform_left(eye(4,4)),
        grasper_approach_parameters_right(4, 0.0), grasper_approach_parameters_left(4, 0.0),
        position_error_threshold(0.01)
    {
        planar_obstacle[2] = 1;
        planar_obstacle[3] = -(-0.15);
    }

};

int main(int argc, char *argv[])
{
    Network yarp;
    ResourceFinder rf;
    rf.setDefaultContext("grasp-processor");
    rf.setDefaultConfigFile("config-icub.ini");
    rf.configure(argc, argv);

    if (!yarp.checkNetwork())
    {
        yError() << prettyError(__FUNCTION__, "YARP network not detected. Check nameserver");
        return EXIT_FAILURE;
    }

    GraspProcessorModule disp;

    return disp.runModule(rf);
}
