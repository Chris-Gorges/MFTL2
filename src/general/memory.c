/*=====================================================================================
 *
 *        Filename:  general.c
 *
 *     Description:  This file contains general functions.
 *
 *          Author:  Berend van Wachem (BvW), berend.van.wachem@gmail.com
 *       Copyright:  (c) Berend van Wachem.
 *
 *
 *
 * =====================================================================================*/

#include <stdlib.h>
#include "mftl.h"
#include "mftl-functions.h"

MFTL_ErrorCode MFTL_AllocateMemory(MFTL_INT size, void **NewPointer)
{
  if (size > 0)
  {
    //    *NewPointer = malloc(size);
    *NewPointer = calloc(1, size);
    if (*NewPointer == MFTL_NULL)
    {
      char error[] = "Malloc for new memory failed.";
      MFTLThrowError(1, error);
    }
  }
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_FreeMemory(void *Pointer)
{
  free(Pointer);
  return (MFTL_SUCCESS);
}
