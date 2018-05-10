class Color(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def blue(s):
        return "{}{}{}".format(Color.OKBLUE, s, Color.ENDC)

    @staticmethod
    def ul(s):
        return "{}{}{}".format(Color.UNDERLINE, s, Color.ENDC)

    @staticmethod
    def bold(s):
        return "{}{}{}".format(Color.BOLD, s, Color.ENDC)
