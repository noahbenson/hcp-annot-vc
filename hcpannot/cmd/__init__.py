################################################################################
# cmd/__init__.py
#
# Command-line processing and configuration tools for hcpannot scripts.
#
# by Noah C. Benson <nben@uw.edu>


################################################################################
# Default Configuration Options:

# The cache directories:
default_cache_path = '/data/crcns2021/hcpannot-cache'

# What we're running.
# The default raters we are processing over.
default_raters = {
    'ventral': [
        'BrendaQiu',
        'bogengsong',
        'JiyeongHa',
        'lindazelinzhao',
        'nourahboujaber',
        'jennifertepan',
        'mean'],
    'dorsal': [
        'Annie-lsc',
        'BrendaQiu',
        'oadesiyan',
        'mominbashir',
        'sc5992',
        'qiutan6li',
        'mean']}

# The hemispheres we are processing over.
default_hemis = ['lh', 'rh']

# The number of processes we want to use (None for all CPUs).
default_nproc = None

# If we want to skip this step whenever the logfile already exists, we can set
# the overwrite value to False. If this is True, then the export trace functions
# will always be run.
default_overwrite = True


################################################################################
# Configuration Class

# The cmd namespace uses a configuration object whose type is specified here.
class Config:
    """A configuration type for hcpannot commands.

    A configuration object for commands that can be run in the hcpannot library.
    This type is intended to make it easier to configure and run commands that
    use `hcpannot` tools.
    """
    __slots__ = (
        'parser',
        'args'
        '_parsed_args')
    @classmethod
    def build_parser(cls, parser):
        parser.add_argument(
            '-c', '--cache-path',
            default=None,
            action='store',
            dest='cache_path',
            help='The cache path to use for the hcpannot library.')
        parser.add_argument(
            '-n', '--nproc',
            default=None,
            type=int,
            action='store',
            dest='nproc',
            help='The number of processors to use.')
        parser.add_argument(
            '-r', '--raters',
            default=['ventral'],
            nargs='*',
            type=str,
            dest='raters',
            help='The raters to process.')
        parser.add_argument(
            '-H', '--hemis',
            default=default_hemis,
            nargs='*',
            type=str,
            dest='hemis',
            help='The hemispheres to process (default: "lh" and "rh").')
        parser.add_argument(
            '-s', '--sids',
            default=None,
            nargs='*',
            type=int,
            dest='sids',
            help='The IDs of the subjects to process (default: all).')
        parser.add_argument(
            '-o', '--overwrite',
            default=False,
            action='store_const',
            const=True,
            dest='overwrite',
            help='Overwrite existing files.')
    def __init__(self, args=None, **kwargs):
        import numpy as np
        # If args is None, we use the system arguments.
        if args is None:
            import sys
            args = sys.argv[1:]
        # We need to build an argument parser to use:
        from argparse import ArgumentParser
        self.parser = ArgumentParser(**kwargs)
        self.build_parser(self.parser)
        self.args = list(args)
        self._parsed_args = None
    def parsed_args(self):
        """Returns the parsed arguments."""
        if self._parsed_args is None:
            # Now, we parse the arguments; we may need to look at environment
            # variables as well.
            import numpy as np
            from os import cpu_count, environ as env
            import sys
            args = self.parser.parse_args(self.args)
            self._parsed_args = args
            if args.cache_path is None:
                args.cache_path = env.get(
                    'HCPANNOT_CACHE_PATH',
                    default_cache_path)
            if len(args.raters) == 1 and args.raters[0] in default_raters:
                args.raters = default_raters[args.raters[0]]
            if args.sids is None:
                from hcpannot import subject_list
                args.sids = subject_list
            args.sids = np.array(args.sids, dtype=int)
            if args.nproc is None or args.nproc == 0:
                args.nproc = cpu_count()
            elif args.nproc < 0:
                args.nproc = cpu_count() + args.nproc
            if args.nproc < 1:
                args.nproc = 1
            args.opts = dict(
                overwrite=args.overwrite,
                cache_path=args.cache_path,
                nproc=args.nproc)
        return self._parsed_args
    @property
    def raters(self):
        return self.parsed_args().raters
    @property
    def sids(self):
        return self.parsed_args().sids
    @property
    def hemis(self):
        return self.parsed_args().hemis
    @property
    def opts(self):
        return self.parsed_args().opts

# A similar configuration type that also expects paths.
class ConfigInOut(Config):
    @classmethod
    def build_parser(cls, parser):
        Config.build_parser(parser)
        parser.add_argument(
            'load_path',
            nargs='?',
            default=None,
            help='The path from which to read the inputs; . by default.')
        parser.add_argument(
            'save_path',
            nargs='?',
            type=str,
            default='.',
            help='The path to which to save the output; load_path by default.')
    def parsed_args(self):
        if self._parsed_args is None:
            args = Config.parsed_args(self)
            self.opts['save_path'] = args.save_path
            self.opts['load_path'] = args.load_path or args.save_path
        return self._parsed_args
