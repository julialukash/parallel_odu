
class PrintHelper:
    def __init__(self):
        pass

    def print_scores(self, artm_model, model_name, n_iterations, output_file_name):
        string = self.artm_model_to_str(artm_model, model_name, n_iterations)
        string = string + 'perplexity_score = {}\nsparsity_phi_score = {}\nsparsity_theta_score = {}\ntopic_kernel_score = {}\n'\
                 .format(artm_model.score_tracker['perplexity_score'].value,
                         artm_model.score_tracker['sparsity_phi_score'].value,
                         artm_model.score_tracker['sparsity_theta_score'].value,
                         artm_model.score_tracker['topic_kernel_score'].average_size)
        if output_file_name != '':
            output_file = open(output_file_name, 'a')
            output_file.write(string)
            output_file.close()
        else:
            print string


    def print_unicode_list(self, name, list, output_file):
        string = name + ': ' + ' '.join(list)
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
                    .format(model_name, artm_model.cache_theta, artm_model.num_document_passes, artm_model.seed)
        if n_iterations != -1:
            str_model = str_model + ', n_iterations = {}'.format(n_iterations)
        if artm_model.scores.data.has_key('top_tokens_score'):
            str_model = str_model + ', n_top_tokens = {}'.format(artm_model.scores['top_tokens_score'].num_tokens)
        if artm_model.scores.data.has_key('topic_kernel_score'):
            str_model = str_model + ', p_threshold = {}'.format(artm_model.scores['topic_kernel_score'].probability_mass_threshold)
        str_model = str_model + '\n'
        return str_model
    
            
    def print_artm_model(self, artm_model, model_name, n_iterations=-1, output_file=None):
        str_model = self.artm_model_to_str(artm_model, model_name, n_iterations)
        if output_file != None:
            output_file.write(str_model)
        print str_model
        
