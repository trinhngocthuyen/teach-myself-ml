
class Logger:
    # http://misc.flogisoft.com/bash/tip_colors_and_formatting
    AVAILABLE_COLOR_CODES = {
        'black':    '\033[30m',
        'red':      '\033[31m',
        'green':    '\033[32m',
        'yellow':   '\033[33m',
        'blue':     '\033[34m',
        'magenta':  '\033[35m',
        'cyan':     '\033[36m',
        'white':    '\033[97m',
        'ENDC':     '\033[0m',
    }

    def __init__(self, color_codes):
        self.color_codes = color_codes

    @staticmethod
    def default():
        color_codes = {
            'debug': Logger.AVAILABLE_COLOR_CODES['white'],
            'info': Logger.AVAILABLE_COLOR_CODES['green'],
            'warning': Logger.AVAILABLE_COLOR_CODES['yellow'],
            'error': Logger.AVAILABLE_COLOR_CODES['red'],
        }
        return Logger(color_codes)

    @staticmethod
    def _makeup(msg, color_code):
        return color_code + msg + Logger.AVAILABLE_COLOR_CODES['ENDC']

    def debug(self, msg):
        print(self._makeup(msg, self.color_codes['debug']))

    def info(self, msg):
        print(self._makeup(msg, self.color_codes['info']))

    def error(self, msg):
        print(self._makeup(msg, self.color_codes['error']))

logger = Logger.default()