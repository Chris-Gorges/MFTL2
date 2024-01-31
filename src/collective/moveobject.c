/***************************************************
 * moveobject.c
 * Created on Tue Jan 30 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/
#include "mftl.h"
#include "mftl-functions.h"
#include <math.h>

VecD VecD_Add(VecD I_A, VecD I_B)
{
  //Adds 2 (vector over double)s
  VecD Result;

  Result._[0] = I_A._[0] + I_B._[0];
  Result._[1] = I_A._[1] + I_B._[1];
  Result._[2] = I_A._[2] + I_B._[2];

  return (Result);
}

MFTL_DOUBLE Norm_VecD(VecD I)
{
  MFTL_DOUBLE Result = 0;

  Result += I._[0] * I._[0];
  Result += I._[1] * I._[1];
  Result += I._[2] * I._[2];

  return (Result);
}

MFTL_DOUBLE Norm_QuadD(QuatD I)
{
  MFTL_DOUBLE Result = 0;

  Result += I._[0] * I._[0];
  Result += I._[1] * I._[1];
  Result += I._[2] * I._[2];
  Result += I._[3] * I._[3];

  return (Result);
}

QuatD QuatD_Multiply_ByScalar(QuatD I_q, MFTL_DOUBLE scalar)
{
  QuatD Result = I_q;

  Result._[0] *= scalar;
  Result._[1] *= scalar;
  Result._[2] *= scalar;
  Result._[3] *= scalar;

  return (Result);
}

MFTL_ErrorCode Multipy_pQuatD_ByScalar(QuatD *target, MFTL_DOUBLE scalar)
{
  target->_[0] *= scalar;
  target->_[1] *= scalar;
  target->_[2] *= scalar;
  target->_[3] *= scalar;
  return (MFTL_SUCCESS);
}

QuatD QuatD_Multiply(QuatD I_pre, QuatD I_post)
{
  const MFTL_DOUBLE qw = I_pre._[0] * I_post._[0] - I_pre._[1] * I_post._[1] - I_pre._[2] * I_post._[2] - I_pre._[3] * I_post._[3];
  const MFTL_DOUBLE qx = I_pre._[0] * I_post._[1] + I_pre._[1] * I_post._[0] + I_pre._[2] * I_post._[3] - I_pre._[3] * I_post._[2];
  const MFTL_DOUBLE qy = I_pre._[0] * I_post._[2] - I_pre._[1] * I_post._[3] + I_pre._[2] * I_post._[0] + I_pre._[3] * I_post._[1];
  const MFTL_DOUBLE qz = I_pre._[0] * I_post._[3] + I_pre._[1] * I_post._[2] - I_pre._[2] * I_post._[1] + I_pre._[3] * I_post._[0];
  QuatD Result = {._ = {qw, qx, qy, qz}};

  return (Result);
}

QuatD QuatD_Conj(QuatD I)
{
  QuatD Result = {._ = {I._[0], -1.0 * I._[1], -1.0 * I._[2], -1.0 * I._[3]}};
  return (Result);
}

MFTL_ErrorCode ReNorm_QuatD(QuatD *target)
{
  MFTL_DOUBLE norm_inv = 1.0;
  MFTL_DOUBLE norm;
  norm = Norm_QuadD(*target);
  if (norm < 1.0e-50)
  {
    //normal of very small is unit
    target->_[0] = 1.0;
    target->_[1] = 0.0;
    target->_[2] = 0.0;
    target->_[3] = 0.0;
  }
  else
  {
    //else divide by the sqrtroot of the norm
    norm_inv /= sqrt(norm);
    Multipy_pQuatD_ByScalar(target, norm_inv);
  }
  return (MFTL_SUCCESS);
}

QuatD QuatD_Inv(QuatD I)
{
  QuatD Result;

  Result = QuatD_Multiply_ByScalar(QuatD_Conj(I), 1.0 / Norm_QuadD(I));

  return (Result);
}

QuatD QuatD_FromW(VecD w_in, MFTL_DOUBLE f_timestep)
{
  MFTL_DOUBLE norm_w;
  MFTL_DOUBLE root_norm_w;
  MFTL_DOUBLE inv_root_norm_w;
  VecD axis;
  MFTL_DOUBLE angle;
  MFTL_DOUBLE sin_half_angle;
  QuatD Result;
  norm_w = Norm_VecD(w_in);
  if (norm_w < 1.0e-40)
  {
    //Very slow rotation, axis not well defined , use 2nd order accurate method
    Result._[0] = 1.0 - 0.5 * (0.5 * f_timestep) * (0.5 * f_timestep) * norm_w;
    Result._[1] = w_in._[0] * 0.5 * f_timestep;
    Result._[2] = w_in._[1] * 0.5 * f_timestep;
    Result._[3] = w_in._[2] * 0.5 * f_timestep;
  }
  else
  {
    //Proper rotation, axis of rotation well defined
    root_norm_w = sqrt(norm_w);
    inv_root_norm_w = 1.0 / root_norm_w;
    axis._[0] = w_in._[0] * inv_root_norm_w;
    axis._[1] = w_in._[1] * inv_root_norm_w;
    axis._[2] = w_in._[2] * inv_root_norm_w;

    angle = root_norm_w * f_timestep;
    sin_half_angle = sin(0.5 * angle);

    Result._[0] = cos(0.5 * angle);
    Result._[1] = axis._[0] * sin_half_angle;
    Result._[2] = axis._[1] * sin_half_angle;
    Result._[3] = axis._[2] * sin_half_angle;
  }

  return (Result);
}

VecD ReframeUI(QuatD Q, QuatD Q_inv, VecD vector)
{
  VecD Result;

  QuatD temp;
  QuatD vect; /*Initially the input vector as a quaternion then the output vector, as a quaternion */

  vect._[0] = 0;
  vect._[1] = vector._[0];
  vect._[2] = vector._[1];
  vect._[3] = vector._[2];
  temp = QuatD_Multiply(Q, vect);
  vect = QuatD_Multiply(temp, Q_inv);

  //Output
  Result._[0] = vect._[1];
  Result._[1] = vect._[2];
  Result._[2] = vect._[3];

  return (Result);
}

QuatD QuatD_From_angles(VecD angles)
{
  MFTL_DOUBLE norm_w;
  MFTL_DOUBLE root_norm_w;
  MFTL_DOUBLE inv_root_norm_w;
  VecD axis;
  MFTL_DOUBLE angle;
  MFTL_DOUBLE sin_half_angle;
  QuatD Result;
  norm_w = Norm_VecD(angles);
  if (norm_w < 1.0e-40)
  {
    //Very slow rotation, axis not well defined , use 2nd order accurate method
    Result._[0] = 1.0 - (0.5 * norm_w);
    Result._[1] = angles._[0];
    Result._[2] = angles._[1];
    Result._[3] = angles._[2];
  }
  else
  {
    //Proper rotation, axis of rotation well defined
    root_norm_w = sqrt(norm_w);
    inv_root_norm_w = 1.0 / root_norm_w;
    axis._[0] = angles._[0] * inv_root_norm_w;
    axis._[1] = angles._[1] * inv_root_norm_w;
    axis._[2] = angles._[2] * inv_root_norm_w;

    angle = root_norm_w;
    sin_half_angle = sin(0.5 * angle);

    Result._[0] = cos(0.5 * angle);
    Result._[1] = axis._[0] * sin_half_angle;
    Result._[2] = axis._[1] * sin_half_angle;
    Result._[3] = axis._[2] * sin_half_angle;
  }

  return (Result);
}

MFTL_ErrorCode MFTL_MoveVertex(MFTL_Vertex *Vertex, MFTL_DOUBLE move[3])
{
  Vertex->x[0] += move[0];
  Vertex->x[1] += move[1];
  Vertex->x[2] += move[2];
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_MoveBody(MFTL_Object *Object, MFTL_DOUBLE move[3])
{
  struct MFTL_Vertex *Vertex;

  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    MFTL_MoveVertex(Vertex, move);
  }

  return (MFTL_SUCCESS);
}

/**************************************************/
//(Not a good system)
MFTL_ErrorCode MFTL_RotateVertex(MFTL_Vertex *v, QuatD Q_rot, QuatD Q_inv, VecD Centre)
{
  VecD Offset;
  VecD NewPos;

  Offset._[0] = v->x[0] - Centre._[0];
  Offset._[1] = v->x[1] - Centre._[1];
  Offset._[2] = v->x[2] - Centre._[2];

  NewPos = VecD_Add(Centre, ReframeUI(Q_rot, Q_inv, Offset));
  //NewPos = VecD_Add(Centre, ReframeUI(Q_inv, Q_rot, Offset));

  v->x[0] = NewPos._[0];
  v->x[1] = NewPos._[1];
  v->x[2] = NewPos._[2];
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_RotateBody(MFTL_Object *Object, MFTL_DOUBLE angle[3], MFTL_DOUBLE centre_of_rotation[3])
{
  MFTL_Vertex *Vertex;

  QuatD Q_Rot;
  QuatD Q_Inv;

  VecD ANGLES;
  VecD Centre;

  ANGLES._[0] = angle[0];
  ANGLES._[1] = angle[1];
  ANGLES._[2] = angle[2];

  Centre._[0] = centre_of_rotation[0];
  Centre._[1] = centre_of_rotation[1];
  Centre._[2] = centre_of_rotation[2];

  Q_Rot = QuatD_From_angles(ANGLES);
  Q_Inv = QuatD_Inv(Q_Rot);

  /* Not necessary but no harm */
  ReNorm_QuatD(&Q_Rot);
  ReNorm_QuatD(&Q_Inv);

  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    MFTL_RotateVertex(Vertex, Q_Rot, Q_Inv, Centre);
  }

  return (MFTL_SUCCESS);
}
