import numpy as np
import math

#计算均使用国际标准单位
material_density = {'Ag':10.49e3,'Au':19.32e3, 'SiO2':2.2e3}
Sci_Constant = {'N_A':6.02214076e23,'FaradayConst':96485}
Relative_mass = {'Ag':107.8682e-3,'Au':196.966e-3,'SiO2':60.1e-3} #kg/mol
Diffusion_Number = {'FcMeOH':7.2e-10,'Ferricyanide':7.2e-10} #m2/s

class nanoparticle():
    def __init__(self,radii,material,ms_con):
        self.radii = radii #m
        self.material = material
        self.ms_con = ms_con #kg/m3;mg/mL
        self.volume = (4/3)*math.pi*(self.radii)**3.0 #m3
        self.density = material_density[material] #kg/m3
        self.singlemass = self.density*self.volume #kg
        self.charge = self.density*self.volume/Relative_mass[material]*Sci_Constant['FaradayConst'] #C

#unit: mol L-1
    def ms_to_amount(self):
        am_con = self.ms_con/(self.singlemass*Sci_Constant['N_A']*1e3)
        return am_con

class electrode():
    def __init__(self,material,radii=0):
        self.material = material
        self.radii = radii

    def lctoradi(self,lc,diffusion_number,concentration,transfer_electron=1):
        self.radii = lc/(4*transfer_electron*diffusion_number*Sci_Constant['FaradayConst']*concentration)

if __name__ == '__main__':
    AgNP_1 = nanoparticle(15e-9,'Ag',0.1)
    print(AgNP_1.ms_to_amount())