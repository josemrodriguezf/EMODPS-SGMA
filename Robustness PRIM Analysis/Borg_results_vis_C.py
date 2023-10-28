import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.text import Annotation

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)


def plot_3D(results: object, pelev: int, pazim: int,index:list, save_path: str = None, insights = None, valid_df = None):
    """returns 3D plot

        :param path_results: path to results
        :param pelev: vertical view plot
        :param pazim:  horizontal view plot
        :return: 3D plot
        """

    sns.set_style('ticks')
    sns.set_context('paper', font_scale=1)

    # getting the original colormap using cm.get_cmap() function
    orig_map = plt.cm.get_cmap('viridis')
    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.invert_xaxis();
    ax.invert_zaxis();
    objs = ax.scatter3D(results['Average Groundwater Depth'] * 0.3048, 
                        results["Reliance"] * (-100),
                        results['Average Revenue'] * -1,
                        c=(results["5th Percentile Minimum Revenue"]*-1),
                        edgecolor='black', 
                        label="95th Percentile Maximum Depth Change",
                        cmap=reversed_map,
                        s=(((results["95th Percentile Maximum Depth Change"] * 0.3048)/max(results["95th Percentile Maximum Depth Change"] * 0.3048))**3)*300,
                        edgecolors='none', marker="v",
                        alpha=0.4,
                        linewidths=0.1,zorder=1)
    
    if valid_df is not None:
        ax.scatter3D(valid_df['Average Groundwater Depth'] * 0.3048, 
                            valid_df["Reliance"] * (100),
                            valid_df['Average Revenue'] * 1,
                            c=(valid_df["5th Percentile Minimum Revenue"]),
                            label="95th Percentile Maximum Depth Change",
                            cmap=reversed_map,
                            s=(((valid_df["95th Percentile Maximum Depth Change"] * 0.3048)/max(valid_df["95th Percentile Maximum Depth Change"] * 0.3048))**3)*300,
                            edgecolors='red', marker="v",
                            alpha=0.4,
                            linewidths=1,zorder=1)
    
       
        
    if insights is True: 
        
        index1 = index[0]
        index2 = index[1]
        index3 = index[2]
        
        ax.scatter3D(results['Average Groundwater Depth'][index1] * 0.3048, 
                     results["Reliance"][index1] * (-100),
                     results['Average Revenue'][index1] * -1,                            
                            s=(((results["95th Percentile Maximum Depth Change"][index1] * 0.3048)/max(results["95th Percentile Maximum Depth Change"] * 0.3048))**3)*300
                            ,edgecolors='red', marker="v", linewidths=2, facecolor="None",
                            c=(results["5th Percentile Minimum Revenue"][index1]*-1),zorder=5);
        
        
        ax.annotate3D('MaxRev', (results['Average Groundwater Depth'][index1] * 0.3048,
                results["Reliance"][index1] * (-100),
                results['Average Revenue'][index1] * -1),
         xytext=(-70, 40),
         textcoords='offset points',
         arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),
         fontsize =15);
        
        ax.scatter3D(results['Average Groundwater Depth'][index2] * 0.3048, 
                     results["Reliance"][index2] * (-100),
                     results['Average Revenue'][index2] * -1,                            
                            s=(((results["95th Percentile Maximum Depth Change"][index2] * 0.3048)/max(results["95th Percentile Maximum Depth Change"] * 0.3048))**3)*300
                            , edgecolors='red', marker="v", linewidths=2,
                            c=(results["5th Percentile Minimum Revenue"][index2]*-1),facecolor="None",zorder=5);
        
        
        ax.annotate3D('60%Rel', (results['Average Groundwater Depth'][index2] * 0.3048, 
                results["Reliance"][index2] * (-100),
                results['Average Revenue'][index2] * -1),
              xytext=(-70, 40),
              textcoords='offset points',
              arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),
              fontsize =15);

        
        
        ax.scatter3D(results['Average Groundwater Depth'][index3]* 0.3048,
                      results["Reliance"][index3] * (-100),
                      results['Average Revenue'][index3] * -1,                            
                            s=(((results["95th Percentile Maximum Depth Change"][index3] * 0.3048)/max(results["95th Percentile Maximum Depth Change"] * 0.3048))**3)*300
                            , edgecolors='red', marker="v", linewidths=2,
                            c=(results["5th Percentile Minimum Revenue"][index3]*-1),
                            facecolor="None",zorder=5);    
        
        ax.annotate3D('MinDepth', (results['Average Groundwater Depth'][index3] * 0.3048,
                results["Reliance"][index3] * (-100),
                results['Average Revenue'][index3] * -1),
              xytext=(-70, 40),
              textcoords='offset points',
              arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),
              fontsize =15);

        
        
        
    ax.set_xlim(50, max(results['Average Groundwater Depth'] * 0.3048));
    ax.set_zlim(11000,max(results['Average Revenue'] * -1) + 500);
    ax.set_ylim(105,5);
    ax.set_xticks(np.arange(50, max(results['Average Groundwater Depth'] * 0.3048) + 10, 20));
    ax.set_zticks(np.arange(11000, max(results['Average Revenue'] * -1) + 500, 1000));
    ax.set_yticks(np.arange(20,120,20))
    ax.view_init(elev=pelev, azim=pazim);
    ax.patch.set_alpha(0);
    ax.zaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('w');
    ax.xaxis.pane.set_edgecolor('w');
    ax.yaxis.pane.set_edgecolor('w');
    ax.set_ylabel('\n\n\n\nReliability\n Groundwater Depth\n Requirement (%)\n'r'$\blacktriangleleft-----$', fontsize=14);
    ax.set_zlabel('\n\n\n\n\n\n\nAverage Total\nRevenue ($M)\n'r'$----\blacktriangleright$', fontsize=14);
    ax.set_xlabel('\n\n\n\nAverage\nGroundwater Depth (m bls)\n'r'$\blacktriangleleft-----$', fontsize=14);
    ax.text(50, 105, max(results['Average Revenue'] * -1) + 500, r"$Ideal$""\n", color='black', fontsize=14);
    ax.zaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.plot(50, 100, max(results['Average Revenue'] * -1) + 500, '*', markersize=20, color="black");
    plt.xticks(fontsize=13);
    plt.yticks(fontsize=13);
    ax.tick_params(axis='z', which='major', pad=15)
    
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(13);
    cax = fig.add_axes([0.6, -0.04, 0.26, 0.03]);
    clb = plt.colorbar(objs, cax=cax, orientation="horizontal");
    clb.ax.tick_params(labelsize=13)

    clb.ax.set_title("5th Percentile\nMinimum Revenue\n($M/Year)\n"r"$-----\blacktriangleright$",
                     fontsize=14);
    
    labels_list = objs.legend_elements("sizes", num=8)[1]
    labels = [int(''.join(i for i in x if i.isdigit())) for x in labels_list]
    labels = (np.array(labels)/300)**(1/3)*max(results["95th Percentile Maximum Depth Change"] * 0.3048)
    labels = np.rint(labels)
    labels = labels.astype(int)
    str1 = list(map(str, labels)) 
    
    str1 = list(str1[i] for i in [0,2,8])
    
    hands =   list(objs.legend_elements("sizes", num=8)[0][i] for i in [0,2,8])
    fig.legend(handles=hands, labels = str1, loc="lower left", ncol=5, columnspacing=0.01,
               title="\n\n       95th Percentile\n  Maximum Depth Change\n       (m bls/Year)\n         "r"$\blacktriangleleft-----$",
               edgecolor="white",
               bbox_to_anchor=(0.22, -0.065, 0.5, 0.1),
               fontsize=13,
               title_fontsize=14,
               framealpha=0);
    clb.ax.locator_params(nbins=5);
    
    

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return (fig)

def parallel_robust(results,target):
    
    target_names = np.array(["Not Robust","Robust"])
    # organize the data
    ys = np.dstack([results["5th Percentile Minimum Revenue"]*-1, results["Average Total Revenue"]*-1,
                    results["Reliability"]*-100, results["Average Groundwater Depth"]*0.3048,
                     results["95th Percentile Maximum Depth Change"]*0.3048])[0]
    
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    
    
    #Names axis
    ynames = ["5th Percentile\nMinimum\nRevenue\n(M USD/year)","Average\nRevenue\n(M USD)","Reliability\n(%)","Average\nGroundwater\nDepth\n(m)",
              "95th Percentile\nMaximum\nDepth Change\n(m/year)"]
    
    ymaxs[3], ymins[3] = ymins[3], ymaxs[3]  
    ymaxs[4], ymins[4] = ymins[4], ymaxs[4]  
    ymaxs[2], ymins[2] = 100, 0 
    dys = ymaxs - ymins
    
    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    
    
    fig, host = plt.subplots(figsize=(7.2,4.2))
    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([int(ymins[i]),int(ymaxs[i])])
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
            ax.set_yticks([int(ymins[i]),int(ymaxs[i])])
            if i == 2:
                ax.set_yticks([0,100])      

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=9.5)
    host.tick_params(axis='x', which='major', pad=6)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    # host.set_title('Parallel Coordinates Plot', fontsize=11)
    
    colors = ["darkgray","red"]
    legend_handles = [None for _ in target_names]
    for j in range(ys.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
            
        if target[j] == 0:
            patch = patches.PathPatch(path, facecolor='none', lw=1, alpha=0.4,
                                  edgecolor=colors[target[j]],zorder=0)
        else:
            patch = patches.PathPatch(path, facecolor='none', lw=1, alpha=0.6,
                                  edgecolor=colors[target[j]],zorder=3)
        
        legend_handles[target[j]] = patch
        host.add_patch(patch)

    host.legend(legend_handles, target_names,
                loc='lower center', bbox_to_anchor=(0.5, -0.18),
                ncol=len(target_names), fancybox=True, shadow=True)



    return fig

def Borg_results_time(data, index1, index2, index3,linewidth_p:int, color3, color1="slateblue", color2="b",
                      save_path: str = None, labels = list):

    fig = plt.figure(figsize=(14, 12))

    plt.subplots_adjust(wspace=0.35, hspace=0.3)

    sub1 = fig.add_subplot(5, 2, 3)
    sub1.set_ylabel("Groundwater\nPumping\nRestriction\n(M m³)", fontsize=16)
    sub1.set_title("b)", loc='right', fontsize=17)
    sub1.plot(np.array(data[index1]["GWP"]) * 1233.48 / 1000000.0, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub1.plot(np.array(data[index2]["GWP"]) * 1233.48 / 1000000.0, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub1.plot(np.array(data[index3]["GWP"]) * 1233.48 / 1000000.0, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub1.tick_params(axis='both', labelsize=16)
    sub1.set_xticklabels([])
    sub1.margins(x=0)

    sub2 = fig.add_subplot(5, 2, 5)  # two rows, two columns, second cell
    sub2.set_title("c)", loc='right', fontsize=17)
    sub2.set_ylabel("Groundwater\nPumping\nFee\n($/m³)", fontsize=16)
    sub2.plot(np.array(data[index1]["GWT"])/1233.48, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub2.plot(np.array(data[index2]["GWT"])/1233.48, linestyle='dotted', linewidth=linewidth_p, c=color2,zorder=0)
    sub2.plot(np.array(data[index3]["GWT"])/1233.48, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub2.tick_params(axis='both', labelsize=16)
    sub2.set_ylim(0,0.5);
    sub2.margins(x=0)
    sub2.set_xticklabels([])

    sub3 = fig.add_subplot(5, 2, 9)  # two rows, two columns, second cell
    sub3.set_title("e)", loc='right', fontsize=17)
    sub3.set_ylabel("TotaL Land\nRestriction\n(K Ha)", fontsize=16)
    sub3.plot(np.array(data[index1]["TL"]) * 0.404686 / 1000, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub3.plot(np.array(data[index2]["TL"]) * 0.404686 / 1000, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub3.plot(np.array(data[index3]["TL"]) * 0.404686 / 1000, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub3.tick_params(axis='both', labelsize=16)
    sub3.set_ylim(0, (max(data[index1]["TL"]) * 0.404686 / 1000)+10)
    sub3.margins(x=0)
    sub3.set_xlabel('Time', fontsize=16)  # we already handled the x-label with ax1

    sub4 = fig.add_subplot(5, 2, 7)  # two rows, two columns, second cell
    sub4.set_title("d)", loc='right', fontsize=17)
    sub4.set_ylabel("Perennials\nPlanting\nRestriction\n(K Ha)", fontsize=16)
    sub4.plot(np.array(data[index1]["PL"]) * 0.404686 / 1000, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub4.plot(np.array(data[index2]["PL"]) * 0.404686 / 1000, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub4.plot(np.array(data[index3]["PL"]) * 0.404686 / 1000, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub4.tick_params(axis='both', labelsize=16)
    sub4.set_ylim(0, (max(data[index1]["PL"]) * 0.404686 / 1000)+10)
    sub4.margins(x=0)
    sub4.set_xticklabels([])

    sub5 = fig.add_subplot(5, 2, 1)  # two rows, two colums, combined third and fourth cell
    sub5.set_ylabel("Surface\nWater\nSupply\n(M m³)", fontsize=16)
    sub5.set_xticklabels([])
    sub5.set_title("\na)", loc='right', fontsize=17)
    sub5.plot(np.array(data[index1]["SW"]) * 1233.48 / 1000000.0, alpha=0.8, linewidth=linewidth_p, c="black")
    # sub5.plot(np.array(data[index2]["SW"]) * 1233.48 / 1000000.0, linestyle='dotted', linewidth=2.5, c=color2)
    sub5.tick_params(axis='both', labelsize=16)
    sub5.margins(x=0)

    sub6 = fig.add_subplot(5, 2, 2)  # two rows, two colums, combined third and fourth cell
    sub6.set_ylabel("Groundwater\nPumping\n(M m³)", fontsize=16)
    sub6.set_xticklabels([])
    sub6.set_title("f)", loc='right', fontsize=17)
    sub6.plot(np.array(data[index1]["Pump_year"]) * 1233.48 / 1000000.0, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub6.plot(np.array(data[index2]["Pump_year"]) * 1233.48 / 1000000.0, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub6.plot(np.array(data[index3]["Pump_year"]) * 1233.48 / 1000000.0, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub6.tick_params(axis='both', labelsize=16)
    sub6.margins(x=0)

    sub7 = fig.add_subplot(5, 2, 4)  # two rows, two colums, combined third and fourth cell
    sub7.set_ylabel("Groundwater\nDepth\n(m bls)", fontsize=16)
    sub7.set_xticklabels([])
    sub7.set_title("g)", loc='right', fontsize=17)
    sub7.plot(np.array(data[index1]["GW_depth_year"]) * 0.3048, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub7.plot(np.array(data[index2]["GW_depth_year"]) * 0.3048, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub7.plot(np.array(data[index3]["GW_depth_year"]) * 0.3048, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub7.plot(np.array([91.6]*30), linestyle='solid', linewidth=linewidth_p*0.9, c="darkorange",alpha=0.5, zorder = 0)
    sub7.tick_params(axis='both', labelsize=16)
    sub7.invert_yaxis()
    sub7.margins(x=0)

    sub8 = fig.add_subplot(5, 2, 6)  # two rows, two colums, combined third and fourth cell
    sub8.set_ylabel("Revenue\n(M $)", fontsize=16)
    sub8.set_xticklabels([])
    sub8.set_title("h)", loc='right', fontsize=17)
    sub8.plot(np.array(data[index1]["Net_revs"]), alpha=0.8, linewidth=linewidth_p, c=color1)
    sub8.plot(np.array(data[index2]["Net_revs"]), linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub8.plot(np.array(data[index3]["Net_revs"]), linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub8.tick_params(axis='both', labelsize=16)
    sub8.set_ylim(0, (max(data[index1]["Net_revs"]))+10)
    sub8.margins(x=0)
    

    sub9 = fig.add_subplot(5, 2, 8)  # two rows, two colums, combined third and fourth cell
    sub9.set_ylabel("Perennial\nCrops Land\n(K Ha)", fontsize=16)
    sub9.set_title("i)", loc='right', fontsize=17)
    sub9.set_xticklabels([])
    sub9.plot(np.array(data[index1]["Perennials_year"]) * 0.404686 / 1000, alpha=0.8, linewidth=linewidth_p, c=color1)
    sub9.plot(np.array(data[index2]["Perennials_year"]) * 0.404686 / 1000, linestyle='dotted', linewidth=linewidth_p, c=color2)
    sub9.plot(np.array(data[index3]["Perennials_year"]) * 0.404686 / 1000, linestyle='dashed', linewidth=linewidth_p, c=color3,alpha=0.8)
    sub9.tick_params(axis='both', labelsize=16)
    sub9.margins(x=0)
    sub9.set_ylim(0, (max(data[index1]["Perennials_year"]) * 0.404686 / 1000)+5)

    sub10 = fig.add_subplot(5, 2, 10)  # two rows, two colums, combined third and fourth cell
    sub10.set_ylabel('Annual\nCrops Land\n(K Ha)', fontsize=16)  # we already handled the x-label with ax1\
    sub10.plot(np.array(np.array(data[index1]["Land_year"])-np.array(data[index1]["Perennials_year"])) * 0.404686 / 1000, linewidth=linewidth_p, c=color1,alpha=0.8,label=labels[0])
    sub10.plot(np.array(np.array(data[index2]["Land_year"])-np.array(data[index2]["Perennials_year"])) * 0.404686 / 1000, linestyle='dotted',linewidth=linewidth_p, c=color2,zorder=0,label=labels[1])
    sub10.plot(np.array(np.array(data[index3]["Land_year"])-np.array(data[index3]["Perennials_year"])) * 0.404686 / 1000, linestyle='dashed',linewidth=linewidth_p, c=color3,alpha=0.8,label=labels[2])
    sub10.set_title("j)", loc='right', fontsize=17)
    sub10.tick_params(axis='both', labelsize=16)
    sub10.margins(x=0)
    sub10.set_xlabel('Time', fontsize=16)  # we already handled the x-label with ax1
    sub10.set_ylim(0, (max(np.array(np.array(data[index1]["Land_year"])-np.array(data[index1]["Perennials_year"]))) * 0.404686 / 1000)+5)
    
    lines_labels = [sub10.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels,loc="upper center",ncol=3,bbox_to_anchor=(0, 0, 1, 0.95),fontsize=16)

    # plt.savefig('add_subplot_1.png', dpi = 300, bbox_inches = 'tight')
    

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return (fig)