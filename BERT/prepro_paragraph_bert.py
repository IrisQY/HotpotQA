from convert import convert_hotpot_to_squad_format
import json

def prepro_bert(data_file):
    json_dict = json.load(open(data_file, 'r'))

    elements = []
    for element in json_dict:
        paragraphs_correct = list(set([facts[0] for facts in element['supporting_facts']]))
        context = [paragraph for paragraph in element['context'] if paragraph[0] in paragraphs_correct]
        element['context'] = context
        elements.append(element)

    with open("paragraph" + data_file, 'w') as fp:
        json.dump(elements, fp)

prepro_bert("hotpot_dev_distractor_v1.json")     
