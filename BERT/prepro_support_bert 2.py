from convert import convert_hotpot_to_squad_format
import json

def prepro_bert(data_file):
    json_dict = json.load(open(data_file, 'r'))

    elements = []
    for element in json_dict:
        dict = {}
        for sentence in element['supporting_facts']:
            paragraph_name = sentence[0]
            pargraph_number = sentence[1]
            if paragraph_name not in dict:
                dict[paragraph_name] = []
            dict[paragraph_name].append(pargraph_number)
        new_context = []
        for context in element['context']:
            if context[0] in dict:
                sentences = [context[1][i] for i in dict[context[0]]]
                new_context_one = [context[0], sentences] 
                new_context.append(new_context_one)

        element['context'] = new_context

        elements.append(element)
        

    with open("support" + data_file, 'w') as fp:
        json.dump(elements, fp)
    print(elements)

prepro_bert("hotpot_dev_distractor_v1.json")
