"""Single command-line entry point for the supported CSI workflow."""
import runpy
import sys

COMMANDS = {
    'web': ('csi.webapp', []),
    'ports': ('csi.ports', []),
    'record': ('csi.recording', []),
    'prepare': ('csi.signal', ['prepare']),
    'plot': ('csi.signal', ['plot']),
    'train': ('csi.training', []),
    'predict': ('csi.realtime', []),
    'monitor': ('csi.viewer', []),
    'audit': ('csi.audit', []),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python tools/run.py <command> [options]\n')
        print('Commands: ' + ', '.join(COMMANDS))
        print('Use <command> --help for command options.')
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f'Unknown command: {command}', file=sys.stderr)
        return 2
    module, prefix = COMMANDS[command]
    sys.argv = [sys.argv[0] if prefix else sys.argv[0] + ' ' + command, *prefix, *sys.argv[2:]]
    runpy.run_module(module, run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
