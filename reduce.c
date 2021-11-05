#include "reduce.h"
#include "params.h"

uint32_t montgomery_reduce(uint64_t a)
{
  uint64_t u;
  u = (a * NEWHOPE_QINV);
  u &= ((1<<NEWHOPE_RLOG)-1);
  u *= NEWHOPE_Q;
  a = a + u;
  a = a >> NEWHOPE_RLOG;
  return a >= NEWHOPE_Q ? (a - NEWHOPE_Q) : a;
}