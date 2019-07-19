import ggplot
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np



class Plotting(object):
    def __init__(self):
        pass

    def conf_setup(self,data_f,confidence, n):
        """ In Labeling Attributes, n = 0 is corpus, n = 1 is word count, n = 2 is symbol count, n = 3 is
        cluster count """
        if data_f == 'list':
            return confidence[n]
        if data_f == 'tuple':
            groups = []
            values = []
            a = confidence[n]
            for i in range(0, len(a)):
                groups.append(a[i][0])
                values.append(a[i][1])
            return groups, values

    def barplot_grouped(self,conf1,conf2,conf3,conf4,attr):
        fig, ax = plt.subplots()

        if attr == 'corpus':
            bar1 = self.conf_setup('tuple', conf1, 0)[1]
            bar2 = self.conf_setup('tuple', conf2, 0)[1]
            bar3 = self.conf_setup('tuple', conf3, 0)[1]
            bar4 = self.conf_setup('tuple', conf4, 0)[1]

            name_group = self.conf_setup('tuple',conf1,0)[0]
            plt.xlabel('Corpus')

        if attr == 'word':
            bar1 = self.conf_setup('list',conf1,1)     # returns ratio of sentences with respect to confidence
            bar2 = self.conf_setup('list',conf2,1)
            bar3 = self.conf_setup('list',conf3,1)
            bar4 = self.conf_setup('list',conf4,1)

            name_group =  ('<= 25','<= 50','<= 100','<= 170','=0')
            plt.xlabel('Word Count')

        if attr == 'symbol':
            bar1 = self.conf_setup('list',conf1,2)     # returns ratio of sentences with respect to confidence
            bar2 = self.conf_setup('list',conf2,2)
            bar3 = self.conf_setup('list',conf3,2)
            bar4 = self.conf_setup('list',conf4,2)

            plt.xlabel('Symbol Count')
            name_group = ('<= 5', '<= 10', '<= 15', '<= 40', '=0')

        if attr == 'cluster':
            bar1 = self.conf_setup('tuple', conf1, 3)[1]
            bar2 = self.conf_setup('tuple', conf2, 3)[1]
            bar3 = self.conf_setup('tuple', conf3, 3)[1]
            bar4 = self.conf_setup('tuple', conf4, 3)[1]

            name_group = self.conf_setup('tuple', conf1, 3)[0]
            plt.xlabel('Cluster')

        bar_w = 0.08  # width of bar
        r1 = np.arange(len(bar1))
        r2 = [x + bar_w for x in r1]
        r3 = [x + bar_w for x in r2]
        r4 = [x + bar_w for x in r3]

        plt.bar(r1, bar1, bar_w, color='blue',  label='100% GG')
        plt.bar(r2, bar2, bar_w, color='cyan', label='66% GG')
        plt.bar(r3, bar3, bar_w, color='red', label='66% NB')
        plt.bar(r4, bar4, bar_w, color='orange', label='100% NB')

        plt.ylabel('Sentence Ratio')
        plt.xticks(r1 + bar_w, name_group)
        plt.legend()

        z = plt.show()



        return z



