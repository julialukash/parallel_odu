import numpy as np
import print_helper as ph
import artm

from os import path, mkdir


def create_model_complex(current_dictionary, n_topics, n_doc_passes, seed_value, n_top_tokens, p_mass_threshold, 
                         common_topics, subject_topics, _debug_print=False):    
    if _debug_print:
        print '[{}] creating model'.format(datetime.now())
    model = artm.ARTM(num_topics=n_topics, dictionary=current_dictionary, cache_theta=True, seed=seed_value, 
                      class_ids={'@default_class': 1.0})
    model.num_document_passes = n_doc_passes
    add_complex_scores_to_model(model, current_dictionary, n_top_tokens=n_top_tokens, p_mass_threshold=p_mass_threshold, 
                                common_topics=common_topics, subject_topics=subject_topics)
    return model

def add_complex_scores_to_model(artm_model, n_top_tokens, p_mass_threshold,
    common_topics, subject_topics, _debug_print=False):
    if _debug_print:
        print '[{}] adding scores'.format(datetime.now())
    # subject
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score_subject', dictionary=dictionary,
                          topic_names=subject_topics))
    artm_model.scores.add(artm.SparsityPhiScore(name='ss_phi_score_subject', class_id='@default_class',
                          topic_names=subject_topics))
    artm_model.scores.add(artm.SparsityThetaScore(name='ss_theta_score_subject',
                          topic_names=subject_topics))
    artm_model.scores.add(artm.TopicKernelScore(name='topic_kernel_score_subject', class_id='@default_class', 
                          topic_names=subject_topics, probability_mass_threshold=p_mass_threshold))
    artm_model.scores.add(artm.TopTokensScore(name='top_tokens_score_subject', class_id='@default_class',
                          topic_names=subject_topics, num_tokens=n_top_tokens))
    
    # common
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score_common', dictionary=dictionary,
                          topic_names=common_topics))
    artm_model.scores.add(artm.SparsityPhiScore(name='ss_phi_score_common', class_id='@default_class',
                          topic_names=common_topics))
    artm_model.scores.add(artm.SparsityThetaScore(name='ss_theta_score_common',
                          topic_names=common_topics))
    artm_model.scores.add(artm.TopicKernelScore(name='topic_kernel_score_common', class_id='@default_class', 
                          topic_names=common_topics, probability_mass_threshold=p_mass_threshold))
    artm_model.scores.add(artm.TopTokensScore(name='top_tokens_score_common', class_id='@default_class', 
                          topic_names=common_topics, num_tokens=n_top_tokens))

def fit_one_model_complex(plot_maker, batch_vectorizer, models_file, config, 
                          model, _n_iterations, _model_name='', _debug_print=False): 
    if _debug_print:
        print '[{}] fitting'.format(datetime.now())
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=_n_iterations)
    if _debug_print:
        print '[{}] outputting'.format(datetime.now())
    ph.print_artm_model(model, _model_name, _n_iterations, output_file=models_file)

    model_pics_file_name =  path.join(config.experiment_path, _model_name)
    plot_maker.make_tm_plots_complex(model, model_pics_file_name)
    
    model_output_file_name = path.join(config.experiment_path, _model_name + '.txt')
    ph.print_scores(model, _model_name, _n_iterations, model_output_file_name)
    ph.print_top_tokens(model, model_output_file_name)
    return model

def create_model_complex(current_dictionary, n_topics, n_doc_passes, seed_value, n_top_tokens, p_mass_threshold, 
                         common_topics, subject_topics, _debug_print=False):    
    if _debug_print:
        print '[{}] creating model'.format(datetime.now())
    model = artm.ARTM(num_topics=n_topics, dictionary=current_dictionary, cache_theta=True, seed=seed_value, 
                      class_ids={'@default_class': 1.0})
    model.num_document_passes = n_doc_passes
    add_complex_scores_to_model(model, n_top_tokens=n_top_tokens, p_mass_threshold=p_mass_threshold, 
                                common_topics=common_topics, subject_topics=subject_topics)
    return model

def add_complex_scores_to_model(artm_model, dictionary, n_top_tokens, p_mass_threshold,
    common_topics, subject_topics, _debug_print=False):
    if _debug_print:
        print '[{}] adding scores'.format(datetime.now())
    # subject
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score_subject', dictionary=dictionary,
                          topic_names=subject_topics))
    artm_model.scores.add(artm.SparsityPhiScore(name='ss_phi_score_subject', class_id='@default_class',
                          topic_names=subject_topics))
    artm_model.scores.add(artm.SparsityThetaScore(name='ss_theta_score_subject',
                          topic_names=subject_topics))
    artm_model.scores.add(artm.TopicKernelScore(name='topic_kernel_score_subject', class_id='@default_class', 
                          topic_names=subject_topics, probability_mass_threshold=p_mass_threshold))
    artm_model.scores.add(artm.TopTokensScore(name='top_tokens_score_subject', class_id='@default_class',
                          topic_names=subject_topics, num_tokens=n_top_tokens))
    
    # common
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score_common', dictionary=dictionary,
                          topic_names=common_topics))
    artm_model.scores.add(artm.SparsityPhiScore(name='ss_phi_score_common', class_id='@default_class',
                          topic_names=common_topics))
    artm_model.scores.add(artm.SparsityThetaScore(name='ss_theta_score_common',
                          topic_names=common_topics))
    artm_model.scores.add(artm.TopicKernelScore(name='topic_kernel_score_common', class_id='@default_class', 
                          topic_names=common_topics, probability_mass_threshold=p_mass_threshold))
    artm_model.scores.add(artm.TopTokensScore(name='top_tokens_score_common', class_id='@default_class', 
                          topic_names=common_topics, num_tokens=n_top_tokens))

def fit_one_model_complex(plot_maker, batch_vectorizer, model, _n_iterations, _model_name='', _debug_print=False): 
    if _debug_print:
        print '[{}] fitting'.format(datetime.now())
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=_n_iterations)
    if _debug_print:
        print '[{}] outputting'.format(datetime.now())
    ph.print_artm_model(model, _model_name, _n_iterations, output_file=models_file)
    
    model_pics_file_name =  path.join(config.experiment_path, _model_name)
    plot_maker.make_tm_plots_complex(model, model_pics_file_name)
    
    model_output_file_name = path.join(config.experiment_path, _model_name + '.txt')
    ph.print_scores(model, _model_name, _n_iterations, model_output_file_name)
    ph.print_top_tokens(model, model_output_file_name)
    return model



def create_model(current_dictionary, n_topics, n_doc_passes, seed_value, n_top_tokens, p_mass_threshold, _debug_print=False):    
    if _debug_print:
        print '[{}] creating model'.format(datetime.now())
    model = artm.ARTM(num_topics=n_topics, dictionary=current_dictionary, cache_theta=True, seed=seed_value, 
                  class_ids={'@default_class': 1.0})
    model.num_document_passes = n_doc_passes
    add_scores_to_model(model, current_dictionary, n_top_tokens=n_top_tokens, p_mass_threshold=p_mass_threshold)
    return model

def add_scores_to_model(artm_model, dictionary, n_top_tokens, p_mass_threshold, _debug_print=False):
    if _debug_print:
        print '[{}] adding scores'.format(datetime.now())
    artm_model.scores.add(artm.PerplexityScore(name='perplexity_score',
                                      dictionary=dictionary))
    artm_model.scores.add(artm.SparsityPhiScore(name='ss_phi_score', class_id='@default_class'))
    artm_model.scores.add(artm.SparsityThetaScore(name='ss_theta_score'))
    artm_model.scores.add(artm.TopicKernelScore(name='topic_kernel_score', class_id='@default_class', 
                                                probability_mass_threshold=p_mass_threshold))
    artm_model.scores.add(artm.TopTokensScore(name='top_tokens_score', class_id='@default_class', num_tokens=n_top_tokens))

def fit_one_model(plot_maker, batch_vectorizer, models_file, config, 
                  model, _n_iterations, _model_name='', _debug_print=False): 
    if _debug_print:
        print '[{}] fitting'.format(datetime.now())
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=_n_iterations)
    if _debug_print:
        print '[{}] outputting'.format(datetime.now())
    ph.print_artm_model(model, _model_name, _n_iterations, output_file=models_file)
    
    model_pics_file_name =  path.join(config.experiment_path, _model_name)
    plot_maker.make_tm_plots(model, model_pics_file_name)
    
    model_output_file_name = path.join(config.experiment_path, _model_name + '.txt')
    ph.print_scores(model, _model_name, _n_iterations, model_output_file_name)
    ph.print_top_tokens(model, model_output_file_name)
    return model