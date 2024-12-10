#%% Comparativa resultados NPs de INIFTA
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy
from datetime import datetime,timedelta
import matplotlib as mpl
from scipy.interpolate import CubicSpline,PchipInterpolator
#%% funciones LECTOR RESULTADOS y  LECTOR CICLOS , calcular Hc
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=18,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

def calcular_hc(H, m):
    # Encuentra los índices donde m cruza el eje x (cambio de signo)
    # el error es la diferencia entre valor negativo y positivo
    cruces = np.where(np.diff(np.sign(m)) != 0)[0]

    hc_valores = []
    for i in cruces:
        # Interpolación lineal para encontrar el cruce exacto
        h1, h2 = H[i], H[i + 1]
        m1, m2 = m[i], m[i + 1]
        h_c = h1 - m1 * (h2 - h1) / (m2 - m1)
        hc_valores.append(h_c)
    
    # Obtén valores positivos y negativos
    hc_positivos = [h for h in hc_valores if h > 0]
    hc_negativos = [h for h in hc_valores if h < 0]

    # Calcula el promedio absoluto de los positivos y negativos
    if hc_positivos and hc_negativos:
        hc_promedio = (np.mean(hc_positivos) + abs(np.mean(hc_negativos))) / 2
        hc_err=    hc_valores[0]+hc_valores[1] 
    else:
        hc_promedio = None  # Si no hay suficientes cruces
    print('Hc = ', hc_promedio,hc_valores)
    return hc_promedio, hc_err


#%% Ciclos y concentracion
ciclos_100 = glob(os.path.join('resultados_100_en_SV', '*ciclo_promedio*'),recursive=True)
ciclos_100.sort()
labels_100 = [os.path.split(s)[-1].split('bobN5')[-1].split('_')[0] for s in ciclos_100]

ciclos_97 = glob(os.path.join('resultados_97_en_SV', '*ciclo_promedio*'),recursive=True)
ciclos_97.sort()
labels_97 = [os.path.split(s)[-1].split('bobN5')[-1].split('_')[0] for s in ciclos_97]

_,_,_,H100_1,M_100_1,meta_100_1=lector_ciclos(ciclos_100[0]) 
_,_,_,H100_2,M_100_2,meta_100_2=lector_ciclos(ciclos_100[1]) 

_,_,_,H97_1,M_97_1,meta_97_1=lector_ciclos(ciclos_97[0]) 
_,_,_,H97_2,M_97_2,meta_97_2=lector_ciclos(ciclos_97[1]) 

 
#%% divido por concentracion
concentracion_100 = np.array([meta_100_1['Concentracion_g/m^3'],
                             meta_100_2['Concentracion_g/m^3']])/1000 #g/L == kg/m³ 

concentracion_97 = np.array([meta_97_1['Concentracion_g/m^3'],
                             meta_97_2['Concentracion_g/m^3']])/1000 #g/L == kg/m³ 

#Divido M por concentracion 
m_100_1 = M_100_1/concentracion_100[0]
m_100_2 = M_100_2/concentracion_100[1]

m_97_1 = M_97_1/concentracion_97[0]
m_97_2 = M_97_2/concentracion_97[1]

#Calculo H coercitivo
print('Coercitivo 100')
Hc_100_1_mean,Hc_100_1_err=calcular_hc(H100_1,m_100_1)
Hc_100_2_mean,Hc_100_2_err=calcular_hc(H100_2,m_100_2)

Hc_100_1=ufloat(Hc_100_1_mean,Hc_100_1_err)
Hc_100_2=ufloat(Hc_100_2_mean,Hc_100_2_err)


print('Coercitivo 97')
Hc_97_1_mean,Hc_97_1_err=calcular_hc(H97_1,m_97_1)
Hc_97_2_mean,Hc_97_2_err=calcular_hc(H97_2,m_97_2)
Hc_97_1=ufloat(Hc_97_1_mean,Hc_97_1_err)
Hc_97_2=ufloat(Hc_97_2_mean,Hc_97_2_err)


#%% Ploteo ciclos 

# Primera figura: ciclos 100 y 97 completos
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), constrained_layout=True,sharey=True)

# Ciclos 100
ax1.plot(H100_1, m_100_1, label=labels_100[0] + f' - {concentracion_100[0]:.2f} g/L')
ax1.plot(H100_2, m_100_2, label=labels_100[1] + f' - {concentracion_100[1]:.2f} g/L', c='tab:red')
# ax1.plot(H100_3, m_100_3, label=labels_100[2] + f' - {concentracion_100[2]:.2f} g/L')
# ax1.plot(H100_4, m_100_4, label=labels_100[3] + f' - {concentracion_100[3]:.2f} g/L')
ax1.set_title('100', fontsize=14)


# Ciclos 97
ax2.plot(H97_1, m_97_1, label=labels_97[0] + f' - {concentracion_97[0]:.2f} g/L')
ax2.plot(H97_2, m_97_2, label=labels_97[1] + f' - {concentracion_97[1]:.2f} g/L', c='tab:red')
# # ax2.plot(H97_3, m_97_3, label=labels_97[2] + f' - {concentracion_97[2]:.2f} g/L')
# # ax2.plot(H97_4, m_97_4, label=labels_97[3] + f' - {concentracion_97[3]:.2f} g/L')
ax2.set_title('97', fontsize=14)

ax1.set_ylabel('M/[NPM] (Am/kg)')
for a in [ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')

plt.savefig('ciclos_100_97.png', dpi=400, facecolor='w')
plt.show()
#%% Segunda figura: zoom en ciclos 100 y 97
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), constrained_layout=True,sharey=True)

ax1.plot(H100_1, m_100_1, '.-', label=labels_100[0] + f' - {Hc_100_1:.0f} A/m')
ax1.plot(H100_2, m_100_2, '.-', label=labels_100[1] + f' - {Hc_100_2:.0f} A/m', c='tab:red')

ax1.set_xlim(-100e2, 100e2)
ax1.set_ylim(-22, 22)
ax1.axhline(0, color='black', linewidth=0.8, zorder=-2)
ax1.axvline(0, color='black', linewidth=0.8, zorder=-2)
ax1.set_title('100', fontsize=14)

# Zoom en ciclos 97
ax2.plot(H97_1, m_97_1, '.-', label=labels_97[0] + f' - {Hc_97_1:.0f} A/m')
ax2.plot(H97_2, m_97_2, '.-', label=labels_97[1] + f' - {Hc_97_2:.0f} A/m', c='tab:red')

ax2.set_xlim(-100e2, 100e2)
ax2.set_ylim(-22, 22)
ax1.set_ylabel('M/[NPM] (Am/kg)')
for a in [ax1,ax2]:
    a.grid()
    a.legend(title='Campo Coercitivo',ncol=2,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    a.axvline(0,0,1,color='k',lw=0.8)
    a.axhline(0,0,1,color='k',lw=0.8)
    a.set_xlabel('H (A/m)')
    
ax2.set_title('97', fontsize=14)

plt.savefig('ciclos_100_97_inset.png', dpi=400, facecolor='w')
plt.show()

#%% Levanto archivos resultados
res_100CP2 = glob(os.path.join('LB100CP2','**','*resultados*'),recursive=True)
res_100OH = glob(os.path.join('LB100OH','**','*resultados*'),recursive=True)
# res_100FP = glob(os.path.join('LB100FP_0.85gL','**','*resultados*'),recursive=True)
# res_100P = glob(os.path.join('LB100P_0.79gL','**','*resultados*'),recursive=True)

res_97CP2 = glob(os.path.join('LB97CP2','**','*resultados*'),recursive=True)
res_97OH = glob(os.path.join('LB97OH','**','*resultados*'),recursive=True)
# res_97FP = glob(os.path.join('LB97FP_0.17gL','**','*resultados*'),recursive=True)
# res_97P = glob(os.path.join('LB97P_0.95gL','**','*resultados*'),recursive=True)
#%% SAR y Tau
(SAR_100CP2,tau_100CP2)=([],[])
(SAR_100OH,tau_100OH)=([],[])
(Hc_100CP2, Hc_100OH) = ([], [])

(SAR_97CP2,tau_97CP2)=([],[])
(SAR_97OH,tau_97OH)=([],[])
(Hc_97CP2, Hc_97OH) = ([], [])

for path in res_100CP2:
    _, _, _, _, Mr, Hc, _, mag_max, xi_M_0, frecuencia_fund, _, _, SAR, tau, _ = lector_resultados(path)
    SAR_100CP2.extend([s for s in SAR if s >= 0])
    tau_100CP2.extend([t for t in tau if t >= 0])
    Hc_100CP2.extend([h for h in Hc if h >= 0])  # Agregar valores válidos de Hc

for path in res_100OH:
    _, _, _, _, Mr, Hc, _, mag_max, xi_M_0, frecuencia_fund, _, _, SAR, tau, _ = lector_resultados(path)
    SAR_100OH.extend([s for s in SAR if s >= 0])
    tau_100OH.extend([t for t in tau if t >= 0])
    Hc_100OH.extend([h for h in Hc if h >= 0])

for path in res_97CP2:
    _, _, _, _, Mr, Hc, _, mag_max, xi_M_0, frecuencia_fund, _, _, SAR, tau, _ = lector_resultados(path)
    SAR_97CP2.extend([s for s in SAR if s >= 0])
    tau_97CP2.extend([t for t in tau if t >= 0])
    Hc_97CP2.extend([h for h in Hc if h >= 0])

for path in res_97OH:
    _, _, _, _, Mr, Hc, _, mag_max, xi_M_0, frecuencia_fund, _, _, SAR, tau, _ = lector_resultados(path)
    SAR_97OH.extend([s for s in SAR if s >= 0])
    tau_97OH.extend([t for t in tau if t >= 0])
    Hc_97OH.extend([h for h in Hc if h >= 0])

#%% Mean & STD
(SAR_100CP2_mean,SAR_100CP2_std) = (np.mean(SAR_100CP2),np.std(SAR_100CP2))
(tau_100CP2_mean,tau_100CP2_std) = (np.mean(tau_100CP2),np.std(tau_100CP2))
(Hc_100CP2_mean, Hc_100CP2_std) = (np.mean(Hc_100CP2), np.std(Hc_100CP2))

(SAR_100OH_mean,SAR_100OH_std) = (np.mean(SAR_100OH),np.std(SAR_100OH))
(tau_100OH_mean,tau_100OH_std) = (np.mean(tau_100OH),np.std(tau_100OH))
(Hc_100OH_mean, Hc_100OH_std) = (np.mean(Hc_100OH), np.std(Hc_100OH))

(SAR_97CP2_mean,SAR_97CP2_std) = (np.mean(SAR_97CP2),np.std(SAR_97CP2))
(tau_97CP2_mean,tau_97CP2_std) = (np.mean(tau_97CP2),np.std(tau_97CP2))
(Hc_97CP2_mean, Hc_97CP2_std) = (np.mean(Hc_97CP2), np.std(Hc_97CP2))

(SAR_97OH_mean,SAR_97OH_std) = (np.mean(SAR_97OH),np.std(SAR_97OH))
(tau_97OH_mean,tau_97OH_std) = (np.mean(tau_97OH),np.std(tau_97OH))
(Hc_97OH_mean, Hc_97OH_std) = (np.mean(Hc_97OH), np.std(Hc_97OH))
#%% Grafico de barras

# Datos

SAR_100CP2= ufloat(SAR_100CP2_mean,SAR_100CP2_std) 
SAR_100OH= ufloat(SAR_100OH_mean,SAR_100OH_std) 

tau_100CP2= ufloat(tau_100CP2_mean,tau_100CP2_std) 
tau_100OH= ufloat(tau_100OH_mean,tau_100OH_std) 

Hc_100CP2 = ufloat(Hc_100CP2_mean, Hc_100CP2_std)
Hc_100OH = ufloat(Hc_100OH_mean, Hc_100OH_std)

SAR_97CP2= ufloat(SAR_97CP2_mean,SAR_97CP2_std) 
SAR_97OH= ufloat(SAR_97OH_mean,SAR_97OH_std) 

tau_97CP2= ufloat(tau_97CP2_mean,tau_97CP2_std) 
tau_97OH= ufloat(tau_97OH_mean,tau_97OH_std) 

Hc_97CP2 = ufloat(Hc_97CP2_mean, Hc_97CP2_std)
Hc_97OH = ufloat(Hc_97OH_mean, Hc_97OH_std)

categories = ['CP2',  'OH']

# Configuración del gráfico
x = [1,1.5]  # Posiciones de las barras
width = 0.3  # Ancho de las barras

#SAR
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4),constrained_layout=True)
bar11 = ax1.bar(x[0], SAR_100CP2_mean, width, yerr=SAR_100CP2_std, capsize=5, color='tab:blue', label=f'{SAR_100CP2:0f} W/g')
bar13 = ax1.bar(x[1], SAR_100OH_mean, width, yerr=SAR_100OH_std, capsize=5, color='tab:orange', label=f'{SAR_100OH:0f} W/g')

bar11 = ax3.bar(x[0], SAR_97CP2_mean, width, yerr=SAR_97CP2_std, capsize=5, color='tab:blue', label=f'{SAR_97CP2:0f} W/g')
bar13 = ax3.bar(x[1], SAR_97OH_mean, width, yerr=SAR_97OH_std, capsize=5, color='tab:orange', label=f'{SAR_97OH:0f} W/g')

for a in[ax1,ax3]:
    a.set_xticks(x)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticklabels(categories)
    a.legend(ncol=1)

ax1.set_title('100',loc='left', fontsize=13)
ax3.set_title('97',loc='left', fontsize=13)
ax1.set_ylabel('SAR (W/g)', fontsize=12)
plt.suptitle('SAR')
plt.savefig('comparativa_SAR_100_97.png',dpi=300)
plt.show()


# #tau 
fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(11, 4),constrained_layout=True)
bar21 = ax2.bar(x[0], tau_100CP2_mean, width, yerr=tau_100CP2_std, capsize=5, color='tab:blue', label=f'{tau_100CP2:0f} ns')
bar23 = ax2.bar(x[1], tau_100OH_mean, width, yerr=tau_100OH_std, capsize=5, color='tab:orange', label=f'{tau_100OH:0f} ns')

bar21 = ax4.bar(x[0], tau_97CP2_mean, width, yerr=tau_97CP2_std, capsize=5, color='tab:blue', label=f'{tau_97CP2:0f} ns')
bar23 = ax4.bar(x[1], tau_97OH_mean, width, yerr=tau_97OH_std, capsize=5, color='tab:orange', label=f'{tau_97OH:0f} ns')

ax2.set_ylabel(r'$\tau$ (ns)', fontsize=12)
ax2.set_title('100',loc='left', fontsize=13)
ax4.set_title('97',loc='left', fontsize=13)

for a in[ax2,ax4]:
    a.set_xticks(x)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticklabels(categories)
    a.legend(ncol=1)
plt.suptitle(r'Tiempo de relajacion $\tau$')    
plt.savefig('comparativa_tau_100_97.png',dpi=300)
plt.show()

#Coercitivo
fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(11, 4),constrained_layout=True)
bar21 = ax2.bar(x[0], Hc_100CP2_mean, width, yerr=Hc_100CP2_std, capsize=5, color='tab:blue', label=f'{Hc_100CP2:0f} kA/m')
bar23 = ax2.bar(x[1], Hc_100OH_mean, width, yerr=Hc_100OH_std, capsize=5, color='tab:orange', label=f'{Hc_100OH:0f} kA/m')

bar21 = ax4.bar(x[0], Hc_97CP2_mean, width, yerr=Hc_97CP2_std, capsize=5, color='tab:blue', label=f'{Hc_97CP2:0f} kA/m')
bar23 = ax4.bar(x[1], Hc_97OH_mean, width, yerr=Hc_97OH_std, capsize=5, color='tab:orange', label=f'{Hc_97OH:0f} kA/m')

ax2.set_ylabel('Hc (kA/m)', fontsize=12)
ax2.set_title('100',loc='left', fontsize=13)
ax4.set_title('97',loc='left', fontsize=13)

for a in[ax2,ax4]:
    a.set_xticks(x)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticklabels(categories)
    a.legend(ncol=1)
plt.suptitle('Campo Coercitivo $H_c$')
plt.savefig('comparativa_Hc_100_97.png',dpi=300)
plt.show()


#%% Armo comparativa de los ciclos promedios de las mismas particulas en agua en VS55
ciclo_100CP2_aq = os.path.join('resultados_100_en_aq', '300kHz_150dA_100Mss_bobN5LB100CP2_ciclo_promedio_H_M.txt')
ciclo_100CP2_SV = os.path.join('resultados_100_en_SV', '300kHz_150dA_100Mss_bobN5LB100CP2VS55_ciclo_promedio_H_M.txt')

ciclo_100OH_aq = os.path.join('resultados_100_en_aq', '300kHz_150dA_100Mss_bobN5LB100OH_ciclo_promedio_H_M.txt')
ciclo_100OH_SV = os.path.join('resultados_100_en_SV', '300kHz_150dA_100Mss_bobN5LB100OHVS55_ciclo_promedio_H_M.txt')

ciclo_97CP2_aq = os.path.join('resultados_97_en_aq', '300kHz_150dA_100Mss_bobN5LB97CP2_ciclo_promedio_H_M.txt')
ciclo_97CP2_SV = os.path.join('resultados_97_en_SV', '300kHz_150dA_100Mss_bobN5LB97CP2VS55_ciclo_promedio_H_M.txt')

ciclo_97OH_aq = os.path.join('resultados_97_en_aq', '300kHz_150dA_100Mss_bobN5LB97OH_ciclo_promedio_H_M.txt')
ciclo_97OH_SV = os.path.join('resultados_97_en_SV', '300kHz_150dA_100Mss_bobN5LB97OHVS55_ciclo_promedio_H_M.txt')


label_100CP2_aq = os.path.split(ciclo_100CP2_aq)[-1].split('bobN5')[-1].split('_')[0]+'_aq'
label_100OH_aq = os.path.split(ciclo_100OH_aq)[-1].split('bobN5')[-1].split('_')[0]+'_aq'
label_100CP2_SV = os.path.split(ciclo_100CP2_SV)[-1].split('bobN5')[-1].split('_')[0][:-4]+'_VS55'
label_100OH_SV = os.path.split(ciclo_100OH_SV)[-1].split('bobN5')[-1].split('_')[0][:-4]+'_VS55'
label_97CP2_aq = os.path.split(ciclo_97CP2_aq)[-1].split('bobN5')[-1].split('_')[0]+'_aq'
label_97OH_aq = os.path.split(ciclo_97OH_aq)[-1].split('bobN5')[-1].split('_')[0]+'_aq'
label_97CP2_SV = os.path.split(ciclo_97CP2_SV)[-1].split('bobN5')[-1].split('_')[0][:-4]+'_VS55'
label_97OH_SV = os.path.split(ciclo_97OH_SV)[-1].split('bobN5')[-1].split('_')[0][:-4]+'_VS55'

#% Leo archivos
_,_,_,H100CP2_aq,M_100CP2_aq,meta_100CP2_aq=lector_ciclos(ciclo_100CP2_aq)
_,_,_,H100CP2_SV,M_100CP2_SV,meta_100CP2_SV=lector_ciclos(ciclo_100CP2_SV)

_,_,_,H100OH_aq,M_100OH_aq,meta_100OH_aq=lector_ciclos(ciclo_100OH_aq)
_,_,_,H100OH_SV,M_100OH_SV,meta_100OH_SV=lector_ciclos(ciclo_100OH_SV)

_,_,_,H97CP2_aq,M_97CP2_aq,meta_97CP2_aq=lector_ciclos(ciclo_97CP2_aq)
_,_,_,H97CP2_SV,M_97CP2_SV,meta_97CP2_SV=lector_ciclos(ciclo_97CP2_SV)

_,_,_,H97OH_aq,M_97OH_aq,meta_97OH_aq=lector_ciclos(ciclo_97OH_aq)
_,_,_,H97OH_SV,M_97OH_SV,meta_97OH_SV=lector_ciclos(ciclo_97OH_SV)

#divido c/u por concentracion 
m_100CP2_aq = M_100CP2_aq/(meta_100CP2_aq[ 'Concentracion_g/m^3']/1000)
m_100CP2_SV = M_100CP2_SV/(meta_100CP2_SV[ 'Concentracion_g/m^3']/1000)
m_100OH_aq = M_100OH_aq/(meta_100OH_aq[ 'Concentracion_g/m^3']/1000)
m_100OH_SV = M_100OH_SV/(meta_100OH_SV[ 'Concentracion_g/m^3']/1000)

m_97CP2_aq = M_97CP2_aq/(meta_97CP2_aq['Concentracion_g/m^3']/1000)
m_97CP2_SV = M_97CP2_SV/(meta_97CP2_SV['Concentracion_g/m^3']/1000)
m_97OH_aq = M_97OH_aq/(meta_97OH_aq['Concentracion_g/m^3']/1000)
m_97OH_SV = M_97OH_SV/(meta_97OH_SV['Concentracion_g/m^3']/1000)

#%% Ploteo ciclos 

# Primera figura: ciclos 100 y 97 completos
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2, figsize=(11, 7), constrained_layout=True,sharey=True)

ax1.plot(H100CP2_aq, m_100CP2_aq, label=label_100CP2_aq + f' - {meta_100CP2_aq["Concentracion_g/m^3"]*1e-3:.2f} g/L')
ax1.plot(H100CP2_SV, m_100CP2_SV, label=label_100CP2_SV + f' - {meta_100CP2_SV["Concentracion_g/m^3"]*1e-3:.2f} g/L')

ax3.plot(H100OH_aq, m_100OH_aq, label=label_100OH_aq + f' - {meta_100OH_aq["Concentracion_g/m^3"]*1e-3:.2f} g/L')
ax3.plot(H100OH_SV, m_100OH_SV, label=label_100OH_SV + f' - {meta_100OH_SV["Concentracion_g/m^3"]*1e-3:.2f} g/L')

ax2.plot(H97CP2_aq, m_97CP2_aq, label=label_97CP2_aq + f' - {meta_97CP2_aq["Concentracion_g/m^3"]*1e-3:.2f} g/L')
ax2.plot(H97CP2_SV, m_97CP2_SV, label=label_97CP2_SV + f' - {meta_97CP2_SV["Concentracion_g/m^3"]*1e-3:.2f} g/L')

ax4.plot(H97OH_aq, m_97OH_aq, label=label_97OH_aq + f' - {meta_97OH_aq["Concentracion_g/m^3"]*1e-3:.2f} g/L')
ax4.plot(H97OH_SV, m_97OH_SV, label=label_97OH_SV + f' - {meta_97OH_SV["Concentracion_g/m^3"]*1e-3:.2f} g/L')

ax1.set_title('100CP2',loc='left', fontsize=14)
ax2.set_title('97CP2',loc='left', fontsize=14)
ax3.set_title('100OH',loc='left', fontsize=14)
ax4.set_title('97OH',loc='left', fontsize=14)

for a in [ax1,ax2,ax3,ax4]:
    a.grid()
    a.legend(ncol=1)

for a in [ax3,ax4]:
    a.set_xlabel('H (A/m)')

for a in [ax1,ax3]:
    a.set_ylabel('M/[NPM] (Am/kg)')

plt.savefig('ciclos_100_97_aq_VS55.png', dpi=400, facecolor='w')
plt.show()
#%%