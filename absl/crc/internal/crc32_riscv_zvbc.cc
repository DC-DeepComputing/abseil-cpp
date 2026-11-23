// Copyright 2026 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/crc/internal/crc32_riscv_zvbc.h"

#if defined(ABSL_CRC_INTERNAL_HAVE_RISCV_ZVBC)

#include "absl/base/config.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace crc_internal {
namespace riscv_zvbc {

#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t crc_bitwise(uint32_t crc,const uint8_t* buf,size_t len){
  uint32_t c=crc;for(size_t i=0;i<len;i++){c^=buf[i];for(int k=0;k<8;k++)c=(c>>1)^((c&1)?0x82F63B78U:0);}return c;
}
static const uint64_t fold_const[4]={0x00000000740eef02ULL,0x000000009e4addf8ULL,0x00000000f20c0dfeULL,0x00000000493c7d27ULL};
#define C0 0x00000000dd45aab8ULL
#define C1 0x00000000493c7d27ULL
#define CQUO 0x0000000dea713f1ULL
#define CPOLY 0x0000000105ec76f1ULL
#define M32 0x00000000ffffffffULL
/* Barrett reduce one 128-bit state (lo,hi) -> 32-bit crc */
static inline uint32_t barrett(uint64_t lo,uint64_t hi){
  uint64_t r; asm volatile("clmul %0,%1,%2":"=r"(r):"r"(lo),"r"((uint64_t)C1));
  uint64_t rh;asm volatile("clmulh %0,%1,%2":"=r"(rh):"r"(lo),"r"((uint64_t)C1));
  uint64_t t1=hi^r, t4=t1&M32; t1>>=32;
  uint64_t t0; asm volatile("clmul %0,%1,%2":"=r"(t0):"r"(t4),"r"((uint64_t)C0));
  uint64_t t3=(rh<<32)^t1^t0;
  t4=t3&M32; asm volatile("clmul %0,%1,%2":"=r"(t4):"r"(t4),"r"((uint64_t)CQUO));
  t4&=M32; asm volatile("clmul %0,%1,%2":"=r"(t4):"r"(t4),"r"((uint64_t)CPOLY));
  t4^=t3;
  return (uint32_t)((t4>>32)&M32);
}
/* fold 4 lanes: (vlo,vhi) = fold((vlo,vhi), k0,k1) xor data (dlo,dhi) */
static inline void vfold(vuint64m2_t* vlo,vuint64m2_t* vhi,uint64_t k0,uint64_t k1,
                          vuint64m2_t dlo,vuint64m2_t dhi){
  vuint64m2_t k0v=__riscv_vmv_v_x_u64m2(k0,4),k1v=__riscv_vmv_v_x_u64m2(k1,4);
  vuint64m2_t lo=__riscv_vclmul_vv_u64m2(*vlo,k0v,4);
  lo=__riscv_vxor_vv_u64m2(lo,__riscv_vclmul_vv_u64m2(*vhi,k1v,4),4);
  /* clmulh via vclmul with shifted? Zvbc has vclmulh? check */
  vuint64m2_t hi=__riscv_vclmulh_vv_u64m2(*vlo,k0v,4);
  hi=__riscv_vxor_vv_u64m2(hi,__riscv_vclmulh_vv_u64m2(*vhi,k1v,4),4);
  *vlo=__riscv_vxor_vv_u64m2(lo,dlo,4);
  *vhi=__riscv_vxor_vv_u64m2(hi,dhi,4);
}
uint32_t Extend(uint32_t crc, const char* buf, size_t len){
  crc^=0xffffffffU;
  const uint8_t* p=reinterpret_cast<const uint8_t*>(buf);size_t n=len;
  size_t pre=(16-((uintptr_t)p&15))&15;if(pre>n)pre=n;
  uint32_t c=crc_bitwise(crc,p,pre);p+=pre;n-=pre;
  if(n<64)return crc_bitwise(c,p,n)^0xffffffffU;
  uint64_t deint[8];memcpy(deint,p,64);
  uint64_t first[4]={deint[0],deint[2],deint[4],deint[6]};
  uint64_t second[4]={deint[1],deint[3],deint[5],deint[7]};
  first[0]^=(uint64_t)c;
  vuint64m2_t vlo=__riscv_vle64_v_u64m2(first,4);
  vuint64m2_t vhi=__riscv_vle64_v_u64m2(second,4);
  p+=64;n-=64;
  const uint64_t k1=fold_const[0],k2=fold_const[1];
  while(n>=64){
    uint64_t de[8];memcpy(de,p,64);
    uint64_t dl[4]={de[0],de[2],de[4],de[6]},dh[4]={de[1],de[3],de[5],de[7]};
    vuint64m2_t dlo=__riscv_vle64_v_u64m2(dl,4),dhi=__riscv_vle64_v_u64m2(dh,4);
    vfold(&vlo,&vhi,k1,k2,dlo,dhi);
    p+=64;n-=64;
  }
  /* reduce 4 lanes -> 1 (need fold by k3,k4 across lanes: scalar extract) */
  uint64_t lo[4],hi[4];__riscv_vse64(lo,vlo,4);__riscv_vse64(hi,vhi,4);
  uint64_t x0=lo[0],x1=hi[0];
  for(int i=1;i<4;i++){
    uint64_t l,r;asm volatile("clmul %0,%1,%2":"=r"(l):"r"(x0),"r"(fold_const[2]));
    asm volatile("clmulh %0,%1,%2":"=r"(r):"r"(x0),"r"(fold_const[2]));
    uint64_t l2,r2;asm volatile("clmul %0,%1,%2":"=r"(l2):"r"(x1),"r"(fold_const[3]));
    asm volatile("clmulh %0,%1,%2":"=r"(r2):"r"(x1),"r"(fold_const[3]));
    x0=l^l2^lo[i];x1=r^r2^hi[i];
  }
  c=barrett(x0,x1);
  if(n)c=crc_bitwise(c,p,n);
  return c^0xffffffffU;
}

}  // namespace riscv_zvbc
}  // namespace crc_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_CRC_INTERNAL_HAVE_RISCV_ZVBC
