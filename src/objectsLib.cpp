/******************************************************************************
 *                                                                            *
 * Copyright (C) 2018 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file objectsLib.cpp
 * @authors: Fabrizio Bottarel <fabrizio.bottarel@iit.it>
 */

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
#include <vtkCaptionActor2D.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <yarp/sig/all.h>
#include <yarp/math/Math.h>

#include <objectsLib.h>

using namespace std;
using namespace yarp::sig;

/****************************************************************/
vtkSmartPointer<vtkActor> &Object::get_actor()
{
    return vtk_actor;
}

/****************************************************************/
Points::Points(const PointCloud<DataXYZRGBA> &points, const int point_size)
{
    vtk_points=vtkSmartPointer<vtkPoints>::New();
    for (size_t i=0; i<points.size(); i++)
        vtk_points->InsertNextPoint(static_cast<double>(points(i).x), static_cast<double>(points(i).y), static_cast<double>(points(i).z));

    vtk_polydata=vtkSmartPointer<vtkPolyData>::New();
    vtk_polydata->SetPoints(vtk_points);

    vtk_glyphFilter=vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vtk_glyphFilter->SetInputData(vtk_polydata);
    vtk_glyphFilter->Update();

    vtk_mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
    vtk_mapper->SetInputConnection(vtk_glyphFilter->GetOutputPort());

    vtk_actor=vtkSmartPointer<vtkActor>::New();
    vtk_actor->SetMapper(vtk_mapper);
    vtk_actor->GetProperty()->SetPointSize(point_size);
}

/****************************************************************/
void Points::set_points(const PointCloud<DataXYZRGBA> &points)
{
    vtk_points=vtkSmartPointer<vtkPoints>::New();
    for (size_t i=0; i<points.size(); i++)
        vtk_points->InsertNextPoint(static_cast<double>(points(i).x), static_cast<double>(points(i).y), static_cast<double>(points(i).z));

    vtk_polydata->SetPoints(vtk_points);
}

/****************************************************************/
bool Points::set_colors(const PointCloud<DataXYZRGBA> &points)
{
    if (points.size()==vtk_points->GetNumberOfPoints())
    {
        vtk_colors=vtkSmartPointer<vtkUnsignedCharArray>::New();
        vtk_colors->SetNumberOfComponents(3);
        for (size_t i=0; i<points.size(); i++)
        {
            vector<unsigned char> colors = {points(i).r, points(i).g, points(i).b};
            vtk_colors->InsertNextTypedTuple(colors.data());
        }

        vtk_polydata->GetPointData()->SetScalars(vtk_colors);
        return true;
    }
    else
        return false;
}

/****************************************************************/
vtkSmartPointer<vtkPolyData> &Points::get_polydata()
{
    return vtk_polydata;
}

/****************************************************************/
Superquadric::Superquadric(const Vector &r, const double color)
{

    //  Default has radius of 0.5, toroidal off, center at 0.0, scale (1,1,1), size 0.5, phi roundness 1.0, and theta roundness 0.0.
    //  Vector r should contain init parameters, but it is left unused for the time being.
    vtk_superquadric=vtkSmartPointer<vtkSuperquadric>::New();

    vtk_superquadric->SetSize(1.0);

    vtk_sample=vtkSmartPointer<vtkSampleFunction>::New();
    vtk_sample->SetSampleDimensions(50,50,50);
    vtk_sample->SetImplicitFunction(vtk_superquadric);
    vtk_sample->SetModelBounds(0, 0, 0, 0, 0, 0);

    vtk_contours=vtkSmartPointer<vtkContourFilter>::New();
    vtk_contours->SetInputConnection(vtk_sample->GetOutputPort());
    vtk_contours->SetValue(0, 0.0);

    vtk_mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
    vtk_mapper->SetInputConnection(vtk_contours->GetOutputPort());
    vtk_mapper->SetScalarRange(0.0,color);

    vtk_actor=vtkSmartPointer<vtkActor>::New();
    vtk_actor->SetMapper(vtk_mapper);
    vtk_actor->GetProperty()->SetOpacity(0.25);

    vtk_transform = vtkSmartPointer<vtkTransform>::New();
    vtk_transform->Identity();

    set_parameters(r);

    vtk_actor->SetUserTransform(vtk_transform);
}

/****************************************************************/
void Superquadric::set_parameters(const Vector &r)
{
    //  set coefficients of the superquadric
    //  (dimensions (x0 x1 x2)) (exponents (x3 x4)) (center (x5 x6 x7)) (orientation (x8 x9 x10 x11))
    //  suppose x8 as angle, x9 x10 x11 define the rotation axis
    vtk_superquadric->SetScale(r[0], r[1], r[2]);
    vtk_superquadric->SetPhiRoundness(r[3]);
    vtk_superquadric->SetThetaRoundness(r[4]);

    vtk_sample->SetModelBounds(-2*r[0], 2*r[0], -2*r[1], 2*r[1], -2*r[2], 2*r[2]);

    //  center of the superquadric is left to zero, is translated by the vtkTransform
    //  translate and set the pose of the superquadric
    vtk_superquadric->SetCenter(0.0, 0.0, 0.0);
    vtk_superquadric->SetToroidal(0);
    vtk_transform->Identity();
    vtk_transform->Translate(r[5], r[6], r[7]);
    vtk_transform->RotateWXYZ(r[8], r[9], r[10], r[11]);

}

/****************************************************************/
Vector Superquadric::getCenter()
{
    //  return the superquadric center coordinates as a yarp::sig::Vector
    double *center;
    Vector center_vec(3);
    center = vtk_transform->GetPosition();
    for (size_t idx = 0; idx < 3; idx++)
    {
        center_vec(idx) = center[idx];
    }

    return center_vec;
}

/****************************************************************/
Vector Superquadric::getAxesSize()
{
    //  get the superquadric semiaxes size and return as a yarp Vector
    double *axes;
    Vector axes_vec;
    axes = vtk_superquadric->GetScale();
    axes_vec.resize(3);
    for (size_t idx = 0; idx<3; idx++)
    {
        axes_vec(idx) = axes[idx];
    }

    //  scale the axes wrt superquadric size
    using namespace yarp::math;
    axes_vec *= vtk_superquadric->GetSize();

    return axes_vec;
}

/****************************************************************/
Vector Superquadric::getOrientationXYZW()
{
    //  convert axis-angle orientation representation to a yarp::sig::Vector
    //  yarp considers axis angle to be [x y z theta]
    //  vtk considers axis angle to be  [theta x y z]
    double* orientationWXYZ;
    Vector orientationWXYZ_vec(4);

    orientationWXYZ = vtk_transform->GetOrientationWXYZ();

    orientationWXYZ_vec(0) = orientationWXYZ[1];
    orientationWXYZ_vec(1) = orientationWXYZ[2];
    orientationWXYZ_vec(2) = orientationWXYZ[3];
    orientationWXYZ_vec(3) = orientationWXYZ[0];

    //yDebug() << "Superquadric processor: orientation WXYZ orientation is " << orientationWXYZ_vec.toString();

    return orientationWXYZ_vec;
}

/****************************************************************/
Vector Superquadric::getRoundness()
{
    //  get the superquadric roundness parameters and return as a yarp Vector
    Vector roundness_vec(2);

    roundness_vec[0] = vtk_superquadric->GetPhiRoundness();
    roundness_vec[1] = vtk_superquadric->GetThetaRoundness();

    //yDebug() << "Superquadric processor: roundness is " << roundness_vec.toString();

    return roundness_vec;
}

/****************************************************************/
GraspPose::GraspPose() : pose_cost_function(2), pose_transform(4,4), pose_rotation(3,3), pose_translation(3), pose_ax_size(3)
{
    pose_cost_function.zero();
    pose_transform.eye();
    pose_rotation.eye();
    pose_translation.zero();
    pose_ax_size.zero();
    pose_vtk_actor = vtkSmartPointer<vtkAxesActor>::New();
    pose_vtk_transform = vtkSmartPointer<vtkTransform>::New();
    pose_vtk_caption_actor = vtkSmartPointer<vtkCaptionActor2D>::New();
}

/****************************************************************/
bool GraspPose::setHomogeneousTransform(const Matrix &rotation, const Vector &translation)
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

/****************************************************************/
void GraspPose::setvtkTransform(const Matrix &transform)
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

/****************************************************************/
void GraspPose::setvtkActorCaption(const string &caption)
{
    pose_vtk_caption_actor->GetTextActor()->SetTextScaleModeToNone();
    pose_vtk_caption_actor->SetCaption(caption.c_str());
    pose_vtk_caption_actor->BorderOff();
    pose_vtk_caption_actor->LeaderOn();
    //pose_vtk_caption_actor->GetCaptionTextProperty()->SetColor(color.data());
    pose_vtk_caption_actor->GetCaptionTextProperty()->SetFontSize(20);
    pose_vtk_caption_actor->GetCaptionTextProperty()->FrameOff();
    pose_vtk_caption_actor->GetCaptionTextProperty()->ShadowOff();
    pose_vtk_caption_actor->GetCaptionTextProperty()->BoldOff();
    pose_vtk_caption_actor->GetCaptionTextProperty()->ItalicOff();
    pose_vtk_caption_actor->SetAttachmentPoint(pose_translation(0), pose_translation(1), pose_translation(2));
}

/****************************************************************/
yarp::sig::Vector GraspPose::getPose()
{
    //  Return the pose in vector format
    //  x y z gx gy gz theta(radians)
    Vector pose(7, 0.0);

    //  set x y z
    pose.setSubvector(0, pose_translation);

    //  set axis-angle representation of orientation
    pose.setSubvector(3, yarp::math::dcm2axis(pose_rotation));

    return pose;

}

/****************************************************************/
bool GraspPose::operator< (const GraspPose &otherPose) const
{
    return pose_cost_function(0) < otherPose.pose_cost_function(0);
}


