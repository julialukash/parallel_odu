import matplotlib.pyplot as plt
import seaborn as sns

class PlotMaker:
    def __init__(self):
        pass

    def make_perplexity_plot(self, perplexity_values, title='',
                         start_iteration=1, end_iteration=-1):
        x_fig_size = 4
        y_fig_size = 2
        if end_iteration == -1:
            end_iteration = len(perplexity_values)
        sns.set_style("white")
        plt.figure(figsize=(x_fig_size, y_fig_size))
        plt.title(title, fontsize=11)
        plt.ylabel('perplexity', fontsize=12)
        plt.xlabel('iterations', fontsize=12)
        plt.tick_params(axis='both', labelsize=9)
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.grid(axis='x',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        # sns.despine(left=True, bottom=True)
        # plt.xlim(0, 5)
        # plt.ylim(0, 0.08)
        iterations = range(start_iteration, end_iteration)
        values = perplexity_values[start_iteration : end_iteration]
        plt.plot(iterations, values, 'bo-')
        plt.show()


    def make_perplexity_sparsity_plot(self, perplexity_values, sparse_phi_values, sparse_theta_values=[],
                                      model_name='', title='',
                                      start_iteration=0, end_iteration=-1):
        x_fig_size = 4
        y_fig_size = 2
        if end_iteration == -1:
            end_iteration = len(perplexity_values)
        x_values = range(start_iteration, end_iteration)
        y_values_1 = perplexity_values[start_iteration : end_iteration]
        y_values_2 = sparse_phi_values[start_iteration : end_iteration]
        y_values_3 = sparse_theta_values[start_iteration : end_iteration]

        sns.set_style("white")
        sns.set_color_codes()
        fig, ax1 = plt.subplots(figsize=(x_fig_size, y_fig_size))
        plt.title(title, fontsize=11)
        plt.xlabel('iterations', fontsize=9)
        plt.grid(axis='x',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=9)
        plot1 = ax1.plot(x_values, y_values_1, 'bo-', label='perplexity')
        ax1.set_ylabel('perplexity', fontsize=9)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        plot2 = ax2.plot(x_values, y_values_2, 'ro-', label='sparsity phi')
        lines = plot1 + plot2
        if len(y_values_3) != 0:
            plot3 = ax2.plot(x_values, y_values_3, 'mo-', label='sparsity theta')
            lines = lines + plot3
        ax2.set_ylabel('sparsity', fontsize=9)
        ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
        plt.show()
        if model_name != '':
            fig.savefig(model_name + '_pv', transparent=True, bbox_inches='tight', pad_inches=0)


    def make_kernel_size_purity_contrast_plot(self, topic_kernel_score, model_name='', title='',
                                              start_iteration=0, end_iteration=-1):
        x_fig_size = 4
        y_fig_size = 2
        if end_iteration == -1:
            end_iteration = len(topic_kernel_score.average_size)
        x_values = range(start_iteration, end_iteration)
        y_values_1 = topic_kernel_score.average_size[start_iteration : end_iteration]
        y_values_2 = topic_kernel_score.average_purity[start_iteration : end_iteration]
        y_values_3 = topic_kernel_score.average_contrast[start_iteration : end_iteration]

        sns.set_style("white")
        sns.set_color_codes()
        fig, ax1 = plt.subplots(figsize=(x_fig_size, y_fig_size))
        plt.title(title, fontsize=11)
        plt.xlabel('iterations', fontsize=9)
        plt.grid(axis='x',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=9)
        plot1 = ax1.plot(x_values, y_values_1, 'bo-', label='avg kernel size')
        ax1.set_ylabel('avg kernel size', fontsize=9)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        plot2 = ax2.plot(x_values, y_values_2, 'ro-', label='avg purity')
        lines = plot1 + plot2
        if len(y_values_3) != 0:
            plot3 = ax2.plot(x_values, y_values_3, 'mo-', label='avg contrast')
            lines = lines + plot3
        ax2.set_ylabel('purity and contrast', fontsize=9)
        ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
    #     ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0, 1), loc=3, borderaxespad=0.)
        plt.show()
        if model_name != '':
            fig.savefig(model_name + '_ksp', transparent=True, bbox_inches='tight', pad_inches=0)


    def make_tm_plots(self, artm_model, model_name=''):
        self.make_perplexity_sparsity_plot(artm_model.score_tracker['perplexity_score'].value,
                                      artm_model.score_tracker['sparsity_phi_score'].value,
                                      artm_model.score_tracker['sparsity_theta_score'].value,
                                      model_name=model_name)
        self.make_kernel_size_purity_contrast_plot(artm_model.score_tracker['topic_kernel_score'], model_name=model_name)