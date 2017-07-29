import os
import re
import sys
import getopt
from shutil import copyfile
from log import logger

class Backup:
    def __init__(self, fname):
        self.fname = fname

    def save(self):
        copyfile(self.fname, self.fname + '.backup')

    def clean(self):
        os.remove(self.fname + '.backup')

    def restore(self):
        copyfile(self.fname + '.backup', self.fname)


def format_notebook(in_fname, out_fname):
    if not os.path.exists(in_fname):
        logger.error('File %s not exist!' % in_fname)
        return

    # Backup input file, for safety
    backup = Backup(in_fname)
    backup.save()

    try:
        updated_lines = []
        n_codecells = 0
        with open(in_fname, 'r', encoding='utf-8') as f_in:
            lines = [line.rstrip() for line in f_in]
            for line in lines:

                match = re.search('( *"execution_count": )([0-9]+)(.*)', line)
                if match is None:
                    updated_lines.append(line)
                else:
                    n_codecells += 1
                    groups = match.groups()  # Expected: len(groups) == 3
                    updated_line = str('%s%d%s' % (groups[0], n_codecells, groups[-1]))
                    updated_lines.append(updated_line)
        f_in.close()

        with open(out_fname, 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join(updated_lines))
        f_out.close()

        # Everything OK --> Clean backup
        backup.clean()

    except IOError as e:
        logger.error('Error processing file: %s' % e)
        # Restore backup errors occur
        backup.restore()


# ----------------------------------
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["in_file=", "out_file="])
    except getopt.GetoptError:
        logger.error('Usage: python format_notebook.py <inputfile> <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            logger.info('Usage: python format_notebook.py <inputfile> <outputfile>')
            sys.exit()

    # Expected to have 2 arguments. If only 1 argument is passed (in_file), then use it as out_file
    in_fname, out_fname = args[0], args[-1]

    logger.info('Format notebook "%s"\t--> Save to "%s"' % (in_fname, out_fname))
    format_notebook(in_fname, out_fname)

if __name__ == "__main__":
    main(sys.argv[1:])