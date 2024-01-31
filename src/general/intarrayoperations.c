/***************************************************
 * intarrayoperations.c
 * Created on Wed Jan 31 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"
#include "mftl-functions.h"

MFTL_ErrorCode MFTL_IntArraySetValue(MFTL_INT *A, MFTL_INT size, MFTL_INT value)
{
  for (MFTL_INT i = 0; i < size; i++) A[i] = value;
  return (MFTL_SUCCESS);
}

MFTL_INT MFTL_FindIntIntervalArray(MFTL_INT num, MFTL_INT *A, MFTL_INT size)
{
  MFTL_INT i;

  for (i = 0; i < size - 1; i++)
  {
    if (num >= A[i] && num < A[i + 1])
    {
      return (i);
    }
  }
  return (size - 1);
}

MFTL_BOOL MFTL_InIntArrayList(MFTL_INT num, MFTL_INT *A, MFTL_INT size)
{
  MFTL_INT i;

  for (i = 0; i < size; i++)
  {
    if (A[i] == num) return (MFTL_YES);
  }
  return (MFTL_NO);
}