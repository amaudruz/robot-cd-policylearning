
layout_names = ['circle_crossing', 'square_crossing', 'cm_hall', 'cm_hall_oneway', 'line', 'line-td', 'tri-td',  'mixed']

def modify_env_params(env, n_humans=None, layout=None ,mods=None) :
    if n_humans :
        env.human_num = n_humans
    if layout :
        env.test_sim = layout
    if mods :
        env.modify_domain(mods)


