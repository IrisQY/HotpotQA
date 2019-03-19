import json

def merge_predictions(predictions_file, yilun_file, output_name):
    predictions_file = json.load(open(predictions_file, 'r'))
    yilun_file = json.load(open(yilun_file, 'r'))
    output_dict = {"answer":{}, "sp":{}}

    for element in yilun_file:
        output_dict["sp"][element["_id"]] = element['sent2title_ids']
    for element in predictions_file:
        output_dict["answer"][element] = predictions_file[element]

    with open(output_name, 'w') as fp:
        json.dump(output_dict, fp)

merge_predictions("final_distractor/predictions.json", "prediction_bert_dev_distractor_60.9.json", "distractortestparagraph.json")
