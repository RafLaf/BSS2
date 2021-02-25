#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _cadecay_reg(void);
extern void _hh2_reg(void);
extern void _IM_reg(void);
extern void _IT_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," cadecay.mod");
    fprintf(stderr," hh2.mod");
    fprintf(stderr," IM.mod");
    fprintf(stderr," IT.mod");
    fprintf(stderr, "\n");
  }
  _cadecay_reg();
  _hh2_reg();
  _IM_reg();
  _IT_reg();
}
