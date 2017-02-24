import numpy as np
import pandas as pd
import print_helper as ph
import distances_helper as dh 
import artm

from datetime import datetime
from scipy.optimize import minimize
from os import path, mkdir

def calculate_distances(dist_fun, _phi, _phi_other, _debug_print=False):
    if _debug_print:
        print '[{}] take_distances between {} columns and {} columns'.format(datetime.now(), len(_phi.columns), len(_phi_other.columns))
    distances = pd.DataFrame(0, index = _phi.columns, columns=_phi_other.columns)
    for idx, col in enumerate(_phi.columns):
        if _debug_print and idx % 20 == 0:
            print '[{}] column {} / {}'.format(datetime.now(), idx, len(_phi.columns))
        for idx_other, col_other in enumerate(_phi_other.columns):
            distance = dist_fun(_phi[col], _phi_other[col_other])
            distances.iloc[idx, idx_other] = distance
    return distances

def get_optimization_result_one_matrix(dist_fn, jac_dist_fn, phi, distances, n_closest_topics, _debug_print=False):
    opt_results = {}
    for col_idx, col_name in enumerate(phi.columns):
        if _debug_print and col_idx % 20 == 0:
            print '[{}] get_optimization_result for column {} / {}'.format(datetime.now(), col_idx, len(phi.columns))
        column = phi[col_name]
        # delete col from phi
        phi_cut = phi.drop(col_name, axis=1)
        opt_results[col_name] = solve_optimization_problem(dist_fn, jac_dist_fn, column, col_name, phi_cut, distances, 
                                                           n_closest_topics)
    return opt_results

def get_optimization_result(dist_fn, jac_dist_fn, phi, phi_other, distances, n_closest_topics,  _debug_print=False):
    opt_results = {}
    for col_idx, col_name in enumerate(phi.columns):
        if _debug_print and col_idx % 20 == 0:
            print '[{}] get_optimization_result for column {} / {}'.format(datetime.now(), col_idx, len(phi.columns))        
        column = phi[col_name]
        opt_results[col_name] = solve_optimization_problem(dist_fn, jac_dist_fn, column, col_name, phi_other, distances,
                                                           n_closest_topics)
    return opt_results

def solve_optimization_problem(dist_fn, jac_dist_fn, column, column_name, phi, distances, n_closest_topics, max_iter=7, verbose=False):
    max_iter = 50
    phi_columns = phi.columns
    # cut distances by phi columns 
    cut_distances = distances[phi_columns]
    # get n closest topics
    closest_column_names = cut_distances.loc[column_name].sort_values().head(n_closest_topics).index.values
    phi_closest = phi[closest_column_names]
    
    # opt solver
    n_columns = phi_closest.shape[1] 
    bnds = [(0, 1)] * n_columns
    constraints = cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1, 'jac': lambda x: [1] * n_columns})
    opt_fun = lambda x: dist_fn(column, phi_closest.dot(x))
    jac_fun = lambda x: jac_dist_fn(column, phi_closest, x)
    
    is_optimized = False
    it = 0
    while (not is_optimized) and it != max_iter:
        it += 1
        init_x = np.random.uniform(0, 1, (1, n_columns))
        init_x /= np.sum(init_x)
        if jac_dist_fn is not None:
            res = minimize(opt_fun, jac=jac_fun, x0=init_x, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': max_iter, 'disp': verbose})
        else:
            res = minimize(opt_fun, x0=init_x, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': max_iter, 'disp': verbose})
        is_optimized = res.success
    if not is_optimized:
        print('Column {} not optimized'.format(column_name))         
    res['column_names'] = phi_closest.columns
    res['optimized_column'] = column_name
#     res['projection'] = phi_closest.dot(res.x)
#     res['column'] = column
    return res




def filter_convex_hull(phi_convex_hull, get_topics_to_remove_fn, n_closest_topics_count, opt_fun_threshold, max_iteration):
    distances_model_iter = calculate_distances(dh.hellinger_dist, phi_convex_hull, phi_convex_hull)
    iterations_info = []
    for n_iteration in range(max_iteration):
        print('[{}] filtering iteration = {} / {}'.format(datetime.now(), n_iteration + 1, max_iteration))
        # get new opts results
        opt_res_convex_hull_inter = get_optimization_result_one_matrix(dh.hellinger_dist, None, 
                                                                       phi_convex_hull, distances_model_iter, n_closest_topics_count)
        # get topics to remove
        topics_to_remove, not_removed_topics_count = get_topics_to_remove_fn(opt_res_convex_hull_inter, distances_model_iter, 
                                                                             n_closest_topics_count, opt_fun_threshold)
        # update phi convex
        phi_convex_hull = remove_topics_from_phi(phi_convex_hull, topics_to_remove)
        distances_model_iter = remove_topics_from_distances_xy(distances_model_iter, topics_to_remove)
        iterations_info.append({'it': n_iteration,
                                'n_topics_to_remove': len(topics_to_remove),
                                'phi_convex_hull_shape': phi_convex_hull.shape,
                                'removed_topics': topics_to_remove,
                                'not_removed_topics_count': not_removed_topics_count,
#                                 'phi_convex_hull': phi_convex_hull, 
                                'opt_res': opt_res_convex_hull_inter})
        print('[{}] {} topics to remove, {} not_removed_topics_count because close topics, current convex_hull shape = {}'.format(datetime.now(), 
               iterations_info[-1]['n_topics_to_remove'], not_removed_topics_count, iterations_info[-1]['phi_convex_hull_shape']))
        if len(iterations_info) >= 1 and iterations_info[-1]['n_topics_to_remove'] == 0:
            print('[{}] topics to remove not increasing, breaking the for loop'.format(datetime.now()))
            break
    return phi_convex_hull, iterations_info

def build_convex_hull_with_filtering(create_model_fn, get_topics_to_remove_fn, words,
                                     init_convex_hull, start_iteration, n_closest_topics_count, opt_fun_threshold,
                                     max_iteration):
    # init phi of convex hull
    phi_convex_hull = init_convex_hull
    if len(phi_convex_hull) == 0:
        phi_convex_hull = pd.DataFrame(0, index = words, columns=[])
    iterations_info = []
    for n_iteration in range(start_iteration, start_iteration + max_iteration):
        print('[{}] ********** iteration = {} / {}'.format(datetime.now(), n_iteration + 1, start_iteration + max_iteration))
        # build model
        model = create_model_fn(n_iteration)
        phi = model.get_phi()
        # rename phi columns 
        phi.columns = [c + '_{}'.format(n_iteration) for c in phi.columns]
        # add to convex hull
        phi_convex_hull_expanded = pd.concat([phi_convex_hull, phi], axis=1)
        # filter topics 
        phi_convex_hull, iterations_info_filter = filter_convex_hull(phi_convex_hull_expanded, get_topics_to_remove_fn,
                                                                     n_closest_topics_count, opt_fun_threshold, max_iteration=35)
        iterations_info.append({'it': n_iteration,
                                'phi_convex_hull_shape': phi_convex_hull.shape,
                                'phi_convex_hull_columns': phi_convex_hull.columns,
#                                 'phi_convex_hull': phi_convex_hull,
                                'iterations_info_filter': iterations_info_filter
                               })
        print('[{}] current convex_hull shape = {}'.format(datetime.now(), 
               iterations_info[-1]['phi_convex_hull_shape']))
    return phi_convex_hull, iterations_info, []

def build_convex_hull_with_filtering_keep_topics(create_model_fn, get_topics_to_remove_fn, words,
                                                 init_convex_hull, start_iteration, n_closest_topics_count,
                                                 opt_fun_threshold, max_iteration):
    # init phi of convex hull
    phi_convex_hull = init_convex_hull
    if len(phi_convex_hull) == 0:
        phi_convex_hull = pd.DataFrame(0, index = words, columns=[])
    else:
        phi_convex_hull = phi_convex_hull[0]
    iterations_info = []
    phi_convex_hull_expanded = phi_convex_hull
    for n_iteration in range(start_iteration, start_iteration + max_iteration):
        print('[{}] ********** iteration = {} / {}'.format(datetime.now(), n_iteration + 1, start_iteration + max_iteration))
        # build model
        model = create_model_fn(n_iteration)
        phi = model.get_phi()
        # rename phi columns 
        phi.columns = [c + '_{}'.format(n_iteration) for c in phi.columns]
        # add to convex hull
        phi_convex_hull_expanded = pd.concat([phi_convex_hull_expanded, phi], axis=1)
        # filter topics 
        phi_convex_hull, iterations_info_filter = filter_convex_hull(phi_convex_hull_expanded, get_topics_to_remove_fn,
                                                                     n_closest_topics_count, opt_fun_threshold, max_iteration=50)
        iterations_info.append({'it': n_iteration,
                                'phi_convex_hull_shape': phi_convex_hull.shape,
                                'phi_convex_hull_columns': phi_convex_hull.columns,
#                                 'phi_convex_hull': phi_convex_hull,
                                'iterations_info_filter': iterations_info_filter
                               })
        print('[{}] current convex_hull shape = {}'.format(datetime.now(), 
               iterations_info[-1]['phi_convex_hull_shape']))
    return phi_convex_hull, iterations_info, []

def build_convex_hull_delayed_filtering(create_model_fn, get_topics_to_remove_fn, words,
                                        init_convex_hull, start_iteration, n_closest_topics_count,
                                        opt_fun_threshold, max_iteration, filtering_iteration):
    # init phi of convex hull
    phi_convex_hull = init_convex_hull
    if len(phi_convex_hull) == 0:
        phi_convex_hull = pd.DataFrame(0, index = words, columns=[])
    iterations_info, iterations_info_filter_list = [], []
    for n_iteration in range(start_iteration, start_iteration + max_iteration):
        print('[{}] ********** iteration = {} / {}'.format(datetime.now(), n_iteration + 1, start_iteration + max_iteration))
        # build model
        model = create_model_fn(n_iteration)
        phi = model.get_phi()
        # rename phi columns 
        phi.columns = [c + '_{}'.format(n_iteration) for c in phi.columns]
        # add to convex hull
        phi_convex_hull = pd.concat([phi_convex_hull, phi], axis=1)
        # filter topics 
        iterations_info.append({'it': n_iteration,
                                'phi_convex_hull_shape': phi_convex_hull.shape})
        print('[{}] current convex_hull shape = {}'.format(datetime.now(), 
               iterations_info[-1]['phi_convex_hull_shape']))
        if ((n_iteration + 1) % filtering_iteration == 0) or (n_iteration + 1 == start_iteration + max_iteration): 
            phi_convex_hull, iterations_info_filter = filter_convex_hull(phi_convex_hull, get_topics_to_remove_fn,
                                                                         n_closest_topics_count, opt_fun_threshold,
                                                                         max_iteration=150)
            iterations_info_filter_list.append(iterations_info_filter)
    return phi_convex_hull, iterations_info, iterations_info_filter_list



def get_topics_to_remove_by_opt_fun_and_distance(opt_res, distances, n_closest, opt_fun_threshold):
    small_dist_opts = {k:i for k, i in opt_res.iteritems() if i['fun'] < opt_fun_threshold}
    sorted_by_fun = sorted(small_dist_opts.values(), key = lambda opt: opt['fun'])
    topics_to_remove = []
    not_removed_count = 0 
    for opt_res in sorted_by_fun:
        topic_name = opt_res.optimized_column
        # check not close to current topics to remove
        is_close_fn = lambda topic, other_topic: other_topic in distances[topic].sort_values().head(n_closest).index
        is_close_to_topics_to_remove = [is_close_fn(topic_name, t) for t in topics_to_remove]
        is_close_to_topics_to_remove = True in is_close_to_topics_to_remove 
        if not is_close_to_topics_to_remove:
            topics_to_remove.append(topic_name)
        else:
            not_removed_count += 1 
    return topics_to_remove, not_removed_count

def get_topics_to_remove_by_opt_fun(opt_res, distances, n_closest, opt_fun_threshold):
    small_dist_opts = {k:i for k, i in opt_res.iteritems() if i['fun'] < opt_fun_threshold}
    topics_to_remove = [x['optimized_column'] for x in small_dist_opts.values()]
    not_removed_count = 0
    return topics_to_remove, not_removed_count

def get_topics_to_remove_by_opt_fun_single(opt_res, distances, n_closest, opt_fun_threshold):
    small_dist_opts = {k:i for k, i in opt_res.iteritems() if i['fun'] < opt_fun_threshold}
    topics_to_remove = [x['optimized_column'] for x in small_dist_opts.values()][0:1]
    not_removed_count = 0
    return topics_to_remove, not_removed_count


def remove_topics_from_phi(phi, topics_to_remove):
    return phi.drop(topics_to_remove, axis=1)

def remove_topics_from_distances_x(distances, topics_to_remove):
    distances_convex_hull = distances.drop(topics_to_remove, axis=0)
    return distances_convex_hull

def remove_topics_from_distances_xy(distances, topics_to_remove):
    distances_convex_hull = distances.drop(topics_to_remove, axis=1)
    distances_convex_hull = distances_convex_hull.drop(topics_to_remove, axis=0)
    return distances_convex_hull