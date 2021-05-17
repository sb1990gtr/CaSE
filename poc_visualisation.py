# region nice_output
from prettytable import PrettyTable
import sys


def nice_output(_data, num_topics):
    """
    makes a nice output as table
    :param _data:
    :return:
    """
    out = "\n ******* FINAL RESULTS****** \n"
    out += "\n reject Hnull percentage\n"
    out += str(_niceTable_main(_data, 1, num_topics, str_unit="%"))
    out += "\n\n average word count\n"
    out += str(_niceTable_main(_data, 2, num_topics))
    out += "\n\n average words in unknown topic\n"
    out += str(_niceTable_main(_data, 3, num_topics))
    out += "\n\n percentage words in unknown topic\n"
    out += str(_niceTable_mainpercentage(_data, 2, 3, num_topics, str_unit="%"))
    return out


def _niceTable_main(_data, idx, num_topics, str_unit="" ):
    out_table = PrettyTable()
    out_table.field_names = ["Percentage"] + [str(i) + " Topics" for i in num_topics]
    for row in _data:
        column = [str(row[0]) + "%"]
        for entry in row[1]:
            column.append(str(entry[idx]) + str_unit)
        out_table.add_row(column)
    return out_table


def _niceTable_mainpercentage(_data, idx_total, idx_cnt, num_topics,  str_unit="%"):
    out_table = PrettyTable()
    out_table.field_names = ["Percentage"] + [str(i) + " Topics" for i in num_topics]
    for row in _data:
        column = [str(row[0]) + "%"]
        for entry in row[1]:
            column.append(str(round(entry[idx_cnt] * 100 / entry[idx_total])) + str_unit)
        out_table.add_row(column)
    return out_table


def _niceListTables(list):
    _table_topiclist = PrettyTable()
    _table_topiclist.field_names = ["pred. class", "feature", "L'(f)"]

    for prediction in list.items():
        pred_class = prediction[0]
        for feature in prediction[1][:10]:
            _table_topiclist.add_row([pred_class, feature[0], feature[1]])
    return _table_topiclist


def nice_output_topic_explanation(topic_explainer, topic_list, word_list):
    """
    format output for topic explainer
    :param topic_explainer:
    :return:
    """
    _table = PrettyTable()
    _table.field_names = ["word_id", "word", "position", "topics", ]
    for row in zip(range(len(topic_explainer.indexed_string.inverse_vocab))[:10]
            , topic_explainer.indexed_string.inverse_vocab
            , topic_explainer.indexed_string.positions
            , topic_explainer.indexed_string.word_topics
                   ):
        _table.add_row(row)
    complete_string = ""
    if topic_explainer.consider_all_words:
        complete_string += ("The unknown Topic is " + str(len(topic_explainer.topics) - 1))

    complete_string += "\n*** Word to Topic mappings\n"
    complete_string += str(_table)
    complete_string += "\n*** The predicted topics\n"
    complete_string += str(_niceListTables(topic_list))
    complete_string += "\n*** The predicted words\n"
    complete_string += str(_niceListTables(word_list))

    return complete_string


def nice_progressbar(percentage):
    p = percentage * 100
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(p / 5), p))




def make_nice_plot (all_data, Param__number_of_topics_param, Param__topic_colors, all_words_cnt):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.title("Scatter of probabilities to reject H0")
    plt.xlabel(" avg words per topic category")
    plt.ylabel("% of rejecting H0")
    legend = [str(entry) + " Topics" for entry in Param__number_of_topics_param]

    for row in all_data:
        for entry, label, color in zip(row[1], legend, Param__topic_colors):
            # x = ((entry[3] / entry[2]) * 100)  # percentage missing words
            x = row[0] * all_words_cnt / 100 # avg words per topic
            y = entry[1]  # percentage reject H0
            ax.scatter(x, y, c=color, label=label, edgecolors='none')

    #ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.legend(legend)
    ax.grid(True)
    plt.show()
