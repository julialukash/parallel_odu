class PrintHelper:
    def __init__(self):
        pass

    def artm_model_last_scores_to_str(self, artm_model):
        string = 'last_perplexity_score = {}\nlast_sparsity_phi_score = {}\nlast_sparsity_theta_score = {}\n'\
                 .format(artm_model.score_tracker['perplexity_score'].value[-1],
                         artm_model.score_tracker['ss_phi_score'].value[-1],
                         artm_model.score_tracker['ss_theta_score'].value[-1]) \
                         + 'last_topic_kernel_avgsize = {}\nlast_topic_kernel_purity = {}\nlast_topic_kernel_contrast = {}\n'\
                 .format(artm_model.score_tracker['topic_kernel_score'].average_size[-1],
                         artm_model.score_tracker['topic_kernel_score'].average_purity[-1],
                         artm_model.score_tracker['topic_kernel_score'].average_contrast[-1])
        return string

    def print_scores(self, artm_model, model_name, n_iterations, output_file_name):
        string = self.artm_model_to_str(artm_model, model_name, n_iterations)
        string += self.artm_model_last_scores_to_str(artm_model)
        string += 'perplexity_score = {}\nsparsity_phi_score = {}\nsparsity_theta_score = {}\n'\
                 .format(artm_model.score_tracker['perplexity_score'].value,
                         artm_model.score_tracker['ss_phi_score'].value,
                         artm_model.score_tracker['ss_theta_score'].value) \
                        + 'topic_kernel_avgsize = {}\ntopic_kernel_purity = {}\ntopic_kernel_contrast = {}\n'\
                 .format(artm_model.score_tracker['topic_kernel_score'].average_size,
                         artm_model.score_tracker['topic_kernel_score'].average_purity,
                         artm_model.score_tracker['topic_kernel_score'].average_contrast)
        if output_file_name != '':
            output_file = open(output_file_name, 'a')
            output_file.write(string)
            output_file.close()
        else:
            print string

    def unicode_list_to_str(self, name, list):
        return name + ': ' + ' '.join(list)

    def print_unicode_list(self, name, list, output_file):
        string = self.unicode_list_to_str(name, list)
        if output_file != None:
            string = string + '\n'
            output_file.write(string.encode('UTF-8'))
        else:
            print string


    def print_top_tokens(self, artm_model, output_file_name=''):
        saved_top_tokens = artm_model.score_tracker['top_tokens_score'].last_tokens
        output_file = None
        if output_file_name != '':
            output_file = open(output_file_name, 'a')
        for topic_name in artm_model.topic_names:
            self.print_unicode_list(topic_name, saved_top_tokens[topic_name], output_file)
        if output_file != None:
            output_file.close()

    def artm_model_to_str(self, artm_model, model_name, n_iterations=-1):
        str_model = 'name = {}, n_topics = {}, n_doc_passes = {}, seed_value = {}'\
                    .format(model_name, artm_model.num_topics, artm_model.num_document_passes, artm_model.seed)
        if n_iterations != -1:
            str_model += ', n_iterations = {}'.format(n_iterations)
        if artm_model.scores.data.has_key('top_tokens_score'):
            str_model += ', n_top_tokens = {}'.format(artm_model.scores['top_tokens_score'].num_tokens)
        if artm_model.scores.data.has_key('topic_kernel_score'):
            str_model += ', p_threshold = {}'.format(artm_model.scores['topic_kernel_score'].probability_mass_threshold)
        regularizers = ''
        for key in artm_model.regularizers.data.iterkeys():
            regularizers += '\n{}, tau = {}'.format(key, artm_model.regularizers.data[key].tau)
        str_model += regularizers + '\n'
        return str_model
    
            
    def print_artm_model(self, artm_model, model_name, n_iterations=-1, output_file=None):
        str_model = self.artm_model_to_str(artm_model, model_name, n_iterations)
        if output_file != None:
            output_file.write(str_model)
        print str_model
        
    def get_doc_top_topics(model, num_top_topics):
        theta = model.get_theta()
        top_topics = {col: theta.ix[:, col].sort_values(ascending=False).head(num_top_topics) for col in theta.columns}
        return top_topics

    def doc_top_topics_to_str(top_topics):
        str = ''
        for key, value in top_topics.iterkeys():
            values = value.iloc[value.nonzero()[0]]
            if len(values):
                value = ', '.join(['{} : {}'.format(ind, values[ind]) for ind in values.index])
            else:
                value = 'None'
            str += '{} | {}\n'.format(key, value)
        return str

    def distances_to_str_row(self, distances, topic, _n_topics):
        values = distances[topic].sort_values().head(_n_topics)
        value = ', '.join(['{0} : [{1:0.2f}]'.format(values.index[ind], val) for ind, val in enumerate(values)])
        str = 'closest by distance to {} | {}\n'.format(topic, value)
        return str

    def print_optimal_solution(self, _sol, _num_components, _distances=None, _saved_top_tokens=None,  _other_saved_top_tokens=None):
        sorted_x = sorted(zip(_sol.x, _sol.column_names), reverse=True)[0 : _num_components]
        if _distances is not None:
            combination = ', '.join(['{0} : {1:0.2f} [{2:0.2f}]'.format(item[1], item[0], _distances[item[1]][_sol.optimized_column]) for item in sorted_x])
            combination += '\n' + self.distances_to_str_row(_distances, _sol.optimized_column, _n_topics=_num_components)
        else:
            combination = ', '.join(['{0} : {1:0.2f}'.format(item[1], item[0]) for item in sorted_x])
        print '============================'
        print 'fun = {}, optimized = {}'.format(_sol.fun, _sol.success)
        print '{} | {}'.format(_sol.optimized_column, combination)
        if _saved_top_tokens is not None and _other_saved_top_tokens is not None:
            topics_str = self.optimal_solution_topics_to_str(_saved_top_tokens, _other_saved_top_tokens, _sol.optimized_column, sorted_x)
            print topics_str
        print '============================'

    def optimal_solution_topics_to_str(self, _saved_top_tokens, _other_saved_top_tokens, topic_name, sorted_x):
        topics = [item[1] for item in sorted_x]
        str = self.unicode_list_to_str(topic_name, _saved_top_tokens[topic_name]) + '\n'
        str += '\n'.join([self.unicode_list_to_str(name, _other_saved_top_tokens[name]) for name in topics])
        return str