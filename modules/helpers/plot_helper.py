import matplotlib.pyplot as plt
import seaborn as sns
import artm 

class PlotMaker:
    def __init__(self, show_plots=False):
        # Turn interactive plotting off
        self.show_plots = show_plots
        if not self.show_plots:
            plt.ioff()


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
        iterations = range(start_iteration, end_iteration)
        values = perplexity_values[start_iteration : end_iteration]
        plt.plot(iterations, values, 'bo-')
        if self.show_plots:
            plt.show()


    def make_perplexity_sparsity_plot(self, perplexity_values, sparse_phi_values, sparse_theta_values=[],
                                      model_name='', title='', filename_ending='',
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
        if self.show_plots:
            plt.show()
        if model_name != '':
            fig.savefig(model_name + '_pv' + filename_ending, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    def make_kernel_size_purity_contrast_plot(self, topic_kernel_score, model_name='', title='', filename_ending = '',
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
        if self.show_plots:
            plt.show()
        if model_name != '':
            fig.savefig(model_name + '_ksp' + filename_ending, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    def artm_model_to_str(self, artm_model, n_iterations=-1):
        str_model = 'n_topics = {}, n_doc_passes = {}, seed_value = {}'\
                    .format(artm_model.num_topics, artm_model.num_document_passes, artm_model.seed)
        if n_iterations != -1:
            str_model += ', n_iterations = {}'.format(n_iterations)
        for score, val in sorted(artm_model.scores.data.iteritems()):
            if type(val) is artm.scores.TopTokensScore:
                str_model += ', {} = {}'.format(score, val.num_tokens)
            if type(val) is artm.scores.TopicKernelScore:
                str_model += ', {} = {}'.format(score, val.probability_mass_threshold)
        regularizers = ''
        for key in artm_model.regularizers.data.iterkeys():
            regularizers += '\n{}, tau = {}'.format(key, artm_model.regularizers.data[key].tau)
        str_model += regularizers + '\n'
        return str_model
    
    def make_tm_plots(self, artm_model, model_name=''):
        title_str = self.artm_model_to_str(artm_model)
        self.make_perplexity_sparsity_plot(artm_model.score_tracker['perplexity_score'].value,
                                      artm_model.score_tracker['ss_phi_score'].value,
                                      artm_model.score_tracker['ss_theta_score'].value,
                                      title=title_str,
                                      model_name=model_name)
        self.make_kernel_size_purity_contrast_plot(artm_model.score_tracker['topic_kernel_score'],
                                                   title=title_str,
                                                   model_name=model_name)
    
    def make_tm_plots_complex(self, artm_model, model_name=''):
        title_str = self.artm_model_to_str(artm_model)
        # common topics
        self.make_perplexity_sparsity_plot(artm_model.score_tracker['perplexity_score_common'].value,
                                      artm_model.score_tracker['ss_phi_score_common'].value,
                                      artm_model.score_tracker['ss_theta_score_common'].value,
                                      title=title_str, filename_ending = '_common',
                                      model_name=model_name)
        self.make_kernel_size_purity_contrast_plot(artm_model.score_tracker['topic_kernel_score_common'],
                                                   title=title_str, filename_ending = '_common',
                                                   model_name=model_name)
        # subject topics
        self.make_perplexity_sparsity_plot(artm_model.score_tracker['perplexity_score_subject'].value,
                                      artm_model.score_tracker['ss_phi_score_subject'].value,
                                      artm_model.score_tracker['ss_theta_score_subject'].value,
                                      title=title_str, filename_ending = '_subject',
                                      model_name=model_name)
        self.make_kernel_size_purity_contrast_plot(artm_model.score_tracker['topic_kernel_score_subject'],
                                                   title=title_str, filename_ending = '_subject',
                                                   model_name=model_name)