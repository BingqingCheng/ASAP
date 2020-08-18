"""
Tools to analyze clustering results
"""

from collections import Counter

import numpy as np


def output_cluster(prefix, labels, dicttags, tags):
    ofile = open("clustered-" + prefix + ".txt", "w")
    # output
    ofile.write("#! tag cluster_id \n")
    ndict = len(dicttags)
    ndata = len(tags)
    for i in range(ndict):
        ofile.write("%s %d\n" % (dicttags[i], labels[i]))
    for i in range(ndata):
        ofile.write("%s %d\n" % (tags[i], labels[i + ndict]))
    ofile.close()

    return 0


def output_cluster_sort(prefix, labels, dicttags, tags):
    # we sort the clusters as well
    ofile = open("sorted-clustered-" + prefix + ".txt", "w")
    # output
    ofile.write("#! tag cluster_id \n")
    ndict = len(dicttags)
    ndata = len(tags)

    sortlabels = np.stack((range(len(labels)), labels), axis=-1)
    sortlabels = sortlabels[sortlabels[:, 1].argsort()]

    for i, l in sortlabels:
        # print i,l
        if l >= 0 and i < ndict:
            ofile.write("%d %s\n" % (l, dicttags[i]))
        elif l >= 0:
            ofile.write("%d %s\n" % (l, tags[i - ndict]))
    return 0


def get_cluster_size(labels):
    unique_labels = set(labels)
    count = Counter(labels)
    return unique_labels, count


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def array_handling(plist, attribute='mean'):
    """ available attributes:
        mean, sum, min, max, mode, all
    """
    if attribute == 'mean':
        return np.mean(plist)
    elif attribute == 'sum':
        return np.sum(plist)
    elif attribute == 'min':
        return np.amin(plist)
    elif attribute == 'max':
        return np.amax(plist)
    elif attribute == 'mode':
        return most_frequent(plist)
    elif attribute == 'all':
        return plist
    else:
        raise NameError('Attribute not found.')


def get_cluster_properties(labels, properties, attribute='mean'):
    unique_labels = set(labels)

    sortlabels = np.stack((range(len(labels)), labels), axis=-1)
    sortlabels = sortlabels[sortlabels[:, 1].argsort()]
    propertiesdict = {-1: 'noise'}

    ol = -1
    n = 0
    plist = []
    for i, l in sortlabels:
        if l > ol and l >= 0:
            propertiesdict[ol] = array_handling(plist, attribute)
            plist = []
        plist.append(properties[i])
        ol = l
    propertiesdict[ol] = array_handling(plist, attribute)

    return unique_labels, propertiesdict


def get_cluster_weighted_avg_properties(labels, properties, weights):
    unique_labels = set(labels)

    sortlabels = np.stack((range(len(labels)), labels), axis=-1)
    sortlabels = sortlabels[sortlabels[:, 1].argsort()]
    propertiesdict = {-1: 'noise'}

    ol = -1
    n = 0
    plist = []
    wlist = []
    for i, l in sortlabels:
        if l > ol and l >= 0:
            propertiesdict[ol] = np.mean(plist) / np.mean(wlist)
            plist = []
            wlist = []
        plist.append(properties[i] * weights[i])
        wlist.append(weights[i])
        ol = l
    propertiesdict[ol] = np.mean(plist) / np.mean(wlist)

    return unique_labels, propertiesdict
