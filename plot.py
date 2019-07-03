import ggplot
import maplotlib
import seaborn


class plot(object):
    def __init__(self, data):
        self.new_df = data

    def barplot(self, x, y, attr):
