#!/usr/bin/python3
'''
Description:
Utils file which has the analysis methods to parse the data
'''
import sys, os
import math
import numpy as np

sys.path.append(os.getcwd())
myPath = str(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(myPath, os.pardir)))


def plotme(plt, plot_id, plot_name, plot_path='./plots', show_flag=True, pdf_only=True):
    if show_flag:
        print('Showing Plot {}_{}'.format(plot_id, plot_name))
        plt.show()
    else:
        if not pdf_only:
            ax = plt.gca()

            if not os.path.exists('{}/png/'.format(plot_path)):
                os.makedirs('{}/png/'.format(plot_path))
            plt.savefig('{}/png/{}_{}.png'.format(plot_path, plot_id, plot_name), format='png', dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.02)
            if not os.path.exists('{}/pdf/'.format(plot_path)):
                os.makedirs('{}/pdf/'.format(plot_path))
            plt.savefig('{}/pdf/{}_{}.pdf'.format(plot_path, plot_id, plot_name), format='pdf', dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.02)
            # Save it with rasterized points
            ax.set_rasterization_zorder(1)
            if not os.path.exists('{}/eps/'.format(plot_path)):
                os.makedirs('{}/eps/'.format(plot_path))
            plt.savefig('{}/eps/{}-{}.eps'.format(plot_path, plot_id, plot_name), dpi=300, rasterized=True,
                        bbox_inches='tight',
                        pad_inches=0.02)
        else:
            if not os.path.exists('{}/pdf/'.format(plot_path)):
                os.makedirs('{}/pdf/'.format(plot_path))
            plt.savefig('{}/pdf/{}_{}.pdf'.format(plot_path, plot_id, plot_name), format='pdf', dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.02)
        print('Saved Plot {}_{}'.format(plot_id, plot_name))
