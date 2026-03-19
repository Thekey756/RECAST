import json


d = {'Length':['word_length', 'sentence_length'],
     'Format':'format',
     'Keyword':'keyword',
     'Strat_With':'start_with',    # Note: Field name has typo 'Strat_With' instead of 'Start_With', but does not affect usage
     'End_With':'end_with',
     'All_Upper':'all_upper',
     'All_lower':'all_lower',
     'No_Commas':'no_commas',
     }




if __name__=='__main__':
    path = './统计加入的约束数量-区分来源-修正列表问题.json'

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    iii = 0
    for item in data[:5]:
        print(f'Processing item {iii}')
        iii += 1
        add_constraint_from_rule = item["added_constraint_from_rule"]
        # rule_based_constraint_dict = item["rule_based_constraint_dict"]
        rule_evaluate_dict = item["rule_evaluate_dict"]
        for k, v_list in add_constraint_from_rule.items():
            for v in v_list:
                # Distinguish between 'word_length' and 'sentence_length' for Length constraints
                if k == "Length":
                    if 'word' in v and 'sentence' not in v:
                        idx = d['Length'][0] # corresponds to word_length
                    elif 'word' not in v and 'sentence' in v:
                        idx = d['Length'][1] # corresponds to sentence_length
                else:
                    idx = d[k]   # Map constraint type from added_constraint_from_rule to rule_evaluate_dict




                evaluator_of_v = rule_evaluate_dict[idx]    # Retrieves the corresponding validator function and parameters
                print(v)
                print('------------')
                print(evaluator_of_v)
                print()





