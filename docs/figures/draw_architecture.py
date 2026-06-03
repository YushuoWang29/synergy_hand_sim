"""
Generate 4 system architecture figures (v3 - English only for matplotlib)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Fig 1: Layered system architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

colors = {
    'input':'#E3F2FD','io':'#FFF3E0','model':'#E8F5E9',
    'synergy':'#F3E5F5','sim':'#FCE4EC','ui':'#EFEBE9',
}
bc = '#37474F'; fc = '#455A64'

def db(ax,x,y,w,h,c,title,items=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12,rounding_size=0.15",
                                facecolor=c,edgecolor=bc,linewidth=2.5,zorder=2))
    ax.text(x+w/2, y+h-0.3, title, ha='center',va='top',fontsize=14,fontweight='bold',color=bc)
    if items:
        for i,t in enumerate(items):
            ax.text(x+0.25, y+h-0.85-i*0.4, t, fontsize=11, color=fc, va='top')

db(ax,0.8,7.8,3.8,1.4,colors['input'],'Input Sources',
   ['.dxf (CAD drawings)', '.ohd (origami design files)', '.urdf (robot models)'])
db(ax,5.5,7.8,3.2,1.4,colors['io'],'Configuration',
   ['Joint stiffness E', 'Transmission matrix R', 'Material properties'])
db(ax,0.5,5.2,8.5,2.2,colors['model'],'Core Model Pipeline',
   ['OrigamiParser -- DXF parsing + graph construction + face detection',
    'OrigamiDesign -- fold lines / faces / joints / pulleys / holes',
    'OrigamiKinematics -- closed-loop FK (iterative relaxation)',
    'OrigamiToURDF -- URDF + STL mesh export'])
db(ax,0.5,3.0,8.5,1.8,colors['synergy'],'Synergy Control Layer',
   ['AdaptiveSynergy -- soft synergy  q = S*sigma + C*J^T*f',
    'AugmentedAdaptiveSynergy -- hard + soft synergy',
    'DynamicSynergy -- damper-based velocity coupling'])
db(ax,0.5,0.3,4.0,2.3,colors['sim'],'Simulation Backend',
   ['MuJoCo physics (URDF)', 'Pinocchio kinematics', 'Gravity / friction / collision',
    'Real-time joint control'])
db(ax,5.2,0.3,4.0,2.3,colors['ui'],'Interactive UI',
   ['MuJoCoSimulator -- 3D viewer + sliders', 'PinocchioSimulator -- 2D CAD + angles',
    'CADViewer -- fold line selection', 'Keyboard shortcuts'])

# Arrows
def ar(ax,x1,y1,x2,y2,lw=2.5,col='#546E7A',sty='arc3,rad=0.15'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=lw,connectionstyle=sty))
ar(ax,2.7,7.8,2.7,7.4); ar(ax,7.1,7.8,7.1,7.4)
ar(ax,4.75,5.2,4.75,4.8)
ar(ax,2.5,3.0,2.5,2.6); ar(ax,7.0,3.0,7.0,2.6)
ax.annotate('',xy=(2.5,2.9),xytext=(2.5,2.5),
            arrowprops=dict(arrowstyle='->',color='#C62828',lw=1.8,
                           connectionstyle='arc3,rad=-0.35',linestyle='dashed'))
ax.text(2.9,2.6,'feedback',fontsize=10,color='#C62828',style='italic')

leg=[mpatches.Patch(color=colors['input'],label='Input/Config'),
     mpatches.Patch(color=colors['model'],label='Core Model'),
     mpatches.Patch(color=colors['synergy'],label='Synergy'),
     mpatches.Patch(color=colors['sim'],label='Simulation'),
     mpatches.Patch(color=colors['ui'],label='Interactive UI')]
ax.legend(handles=leg,loc='upper right',fontsize=10,framealpha=0.9,edgecolor=bc,ncol=2)
ax.set_title('Synergy Hand Simulator Software Architecture',fontsize=17,fontweight='bold',pad=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'architecture.png'),dpi=200,bbox_inches='tight')
plt.close()
print("  [OK] architecture.png")

# ============================================================
# Fig 2: Workflow
# ============================================================
fig2,ax2 = plt.subplots(1,1,figsize=(13,8))
ax2.set_xlim(0,13); ax2.set_ylim(0,8); ax2.axis('off')

steps1 = [('1. Load DXF','#FF8A65'),('2. Parse Geometry','#FF8A65'),
          ('3. Build Topology','#FFB74D'),('4. Export URDF+STL','#FFB74D')]
for i,(lab,c) in enumerate(steps1):
    x=0.3+i*3.2
    ax2.add_patch(FancyBboxPatch((x,6.2),2.8,0.9,
        boxstyle="round,pad=0.08",facecolor=c,edgecolor=bc,linewidth=2))
    ax2.text(x+1.4,6.65,lab,ha='center',va='center',fontsize=11,fontweight='bold',color='white')
    if i<3:
        ax2.annotate('',xy=(x+3.0,6.65),xytext=(x+2.85,6.65),
                     arrowprops=dict(arrowstyle='->',lw=2.5,color='#546E7A'))

steps2 = [('5. Load MuJoCo','#66BB6A'),('6. Define Synergy','#66BB6A'),
          ('7. Simulate','#4DB6AC'),('8. Analyze Output','#4DB6AC')]
for i,(lab,c) in enumerate(steps2):
    x=0.3+i*3.2
    ax2.add_patch(FancyBboxPatch((x,4.3),2.8,0.9,
        boxstyle="round,pad=0.08",facecolor=c,edgecolor=bc,linewidth=2))
    ax2.text(x+1.4,4.75,lab,ha='center',va='center',fontsize=11,fontweight='bold',color='white')
    if i<3:
        ax2.annotate('',xy=(x+3.0,4.75),xytext=(x+2.85,4.75),
                     arrowprops=dict(arrowstyle='->',lw=2.5,color='#546E7A'))
ax2.annotate('',xy=(10.0,5.7),xytext=(10.0,5.2),
             arrowprops=dict(arrowstyle='->',lw=2.5,color='#546E7A'))
ax2.text(10.3,5.45,'URDF',fontsize=10,color='#546E7A')

# Bottom panels
for x,t,items in [(0.3,'Interactive Adjustment',
    ['Click fold line -> select joint','Drag slider / keyboard','Real-time 3D update']),
    (5.3,'Synergy Demo (Auto)',['sigma_0: root joints','sigma_1: tip joints','Animated playback'])]:
    ax2.add_patch(FancyBboxPatch((x,1.0),4.5,2.0,
        boxstyle="round,pad=0.08",facecolor='#F5F5F5',edgecolor=bc,linewidth=2))
    ax2.text(x+2.25,2.85,t,ha='center',va='top',fontsize=13,fontweight='bold',color=bc)
    for i,ti in enumerate(items):
        ax2.text(x+0.35,2.4-i*0.38,'  '+ti,fontsize=10,color=fc,va='top')

ax2.add_patch(FancyBboxPatch((10.5,1.7),2.2,1.3,
    boxstyle="round,pad=0.08",facecolor='#B39DDB',edgecolor=bc,linewidth=2))
ax2.text(11.6,2.35,'Export\nResults',ha='center',va='center',fontsize=12,fontweight='bold',color='white')

ax2.annotate('',xy=(8.5,3.5),xytext=(9.0,4.3),
             arrowprops=dict(arrowstyle='->',lw=1.5,color='#546E7A'))
ax2.annotate('',xy=(5.5,2.5),xytext=(6.5,4.3),
             arrowprops=dict(arrowstyle='->',lw=1.5,color='#546E7A'))
ax2.annotate('',xy=(10.5,2.35),xytext=(5.0,2.0),
             arrowprops=dict(arrowstyle='->',lw=1.5,color='#546E7A',linestyle='dashed'))
ax2.annotate('',xy=(0.3,2.0),xytext=(0.3,3.5),
             arrowprops=dict(arrowstyle='->',color='#C62828',lw=1.8,
                            connectionstyle='arc3,rad=-0.45',linestyle='dashed'))
ax2.text(0.0,2.8,'adjust\nparams',fontsize=9,color='#C62828',ha='center')

ax2.set_title('Typical Workflow of Synergy Hand Simulator',fontsize=16,fontweight='bold',pad=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'workflow.png'),dpi=200,bbox_inches='tight')
plt.close()
print("  [OK] workflow.png")

# ============================================================
# Fig 3: Synergy model
# ============================================================
fig3,ax3 = plt.subplots(1,1,figsize=(11,7))
ax3.set_xlim(0,11); ax3.set_ylim(0,7.5); ax3.axis('off')

def b3(ax,x,y,w,h,c,ec,txt,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor=c,edgecolor=ec,linewidth=2.5))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,fontweight='bold',color=ec)
def a3(ax,x1,y1,x2,y2,col='#546E7A',lw=2.5,sty='arc3,rad=0.1'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=lw,connectionstyle=sty))

b3(ax3,0.3,3.0,2.0,1.2,'#E3F2FD','#1565C0','sigma\n(synergy input)',12)
b3(ax3,3.0,3.0,2.0,1.2,'#F3E5F5','#7B1FA2','R\n(transmission)',12)
b3(ax3,5.7,3.0,2.0,1.2,'#E8F5E9','#2E7D32','E^{-1}\n(compliance)',12)
b3(ax3,8.2,2.4,2.5,2.4,'#FFF3E0','#E65100',
   'Synergy Matrix S\nS = E^{-1}R^T(RE^{-1}R^T)^{-1}\nq_act = S * sigma',11)
b3(ax3,8.2,0.3,2.5,1.5,'#FCE4EC','#C62828',
   'Compliance C\nC = E^{-1} - SRE^{-1}\nq_pass = C*J^T*f_ext',10)
b3(ax3,0.3,0.3,2.0,1.2,'#B2EBF2','#00838F','Joint angles q\n= q_act + q_pass',11)
b3(ax3,0.3,5.5,2.0,0.8,'#FFEBEE','#C62828','f_ext\n(external force)',10)

a3(ax3,2.3,3.6,3.0,3.6); a3(ax3,5.0,3.6,5.7,3.6); a3(ax3,7.7,3.6,8.2,3.6)
a3(ax3,2.3,6.0,5.7,4.2,col='#C62828')
a3(ax3,6.7,2.4,6.7,2.0,col='#C62828',sty='arc3,rad=-0.2')
ax3.text(6.7,1.6,'q_pass',fontsize=10,color='#C62828',ha='center')
a3(ax3,2.3,1.5,6.7,2.8)
ax3.text(4.5,1.7,'q_act = S*sigma',fontsize=10,color='#E65100',ha='center')
a3(ax3,2.3,5.9,3.0,4.2,col='#C62828')
ax3.text(2.5,5.1,'J^T*f_ext',fontsize=9,color='#C62828')

ax3.set_title('Adaptive Synergy Control Model (Grioli et al. 2012)',fontsize=14,fontweight='bold',pad=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'synergy_model.png'),dpi=200,bbox_inches='tight')
plt.close()
print("  [OK] synergy_model.png")

# ============================================================
# Fig 4: Module dependency
# ============================================================
fig4,ax4 = plt.subplots(1,1,figsize=(13,9))
ax4.set_xlim(0,13); ax4.set_ylim(0,9); ax4.axis('off')

mods=[
    (0.5,5.5,3.0,1.4,'#C5CAE9','origami_design.py\ndata model container'),
    (0.5,3.0,3.0,1.4,'#C5CAE9','origami_parser.py\nDXF parser + graph'),
    (4.0,5.5,3.0,1.4,'#C5CAE9','origami_kinematics.py\nclosed-loop FK'),
    (4.0,3.0,3.0,1.4,'#C5CAE9','origami_to_urdf.py\nURDF + STL export'),
    (8.0,6.2,3.2,1.0,'#F8BBD0','base_adaptive.py\nadaptive synergy'),
    (8.0,4.5,3.2,1.0,'#F8BBD0','augmented_adaptive.py\naugmented synergy'),
    (8.0,2.8,3.2,1.0,'#F8BBD0','dynamic_synergy.py\ndynamic synergy'),
    (0.5,1.0,3.0,1.2,'#B2DFDB','mujoco_simulator.py\nMuJoCo 3D sim'),
    (4.0,1.0,3.0,1.2,'#B2DFDB','pinocchio_simulator.py\nPinocchio sim'),
    (4.0,0.0,3.0,0.8,'#B2DFDB','cad_viewer.py\n2D fold view'),
]
for x,y,w,h,c,txt in mods:
    ax4.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.1",
                                 facecolor=c,edgecolor=bc,linewidth=1.8))
    ax4.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10,fontweight='bold',color=fc)

deps=[(3.5,6.2,4.0,6.2),(3.5,3.7,4.0,3.7),(7.0,6.0,8.0,6.7),
      (7.0,6.0,8.0,5.0),(7.0,6.0,8.0,3.3),(4.0,2.4,3.5,2.2),(4.0,2.4,4.0,2.2)]
for x1,y1,x2,y2 in deps:
    ax4.annotate('',xy=(x2,y2),xytext=(x1,y1),
                 arrowprops=dict(arrowstyle='->',lw=2,color='#78909C'))

ax4.text(2.0,7.8,'Core Models (src/models/)',ha='center',fontsize=13,
         fontweight='bold',color='#5C6BC0')
ax4.text(9.6,7.8,'Synergy Control (src/synergy/)',ha='center',fontsize=13,
         fontweight='bold',color='#E91E63')
ax4.text(2.0,0,'Interactive Simulators (src/interactive/)',ha='center',
         fontsize=13,fontweight='bold',color='#00897B')
ax4.set_title('Code Module Dependency Graph',fontsize=16,fontweight='bold',pad=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'modules.png'),dpi=200,bbox_inches='tight')
plt.close()
print("  [OK] modules.png")

print("\nAll figures v3 generated successfully!")
