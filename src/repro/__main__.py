"""One entry point for evidence checks, saved-score tables, and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'evidence', 'tables', 'figures'))
    parser.add_argument('--output', type=Path, default=Path('reproduced'))
    parser.add_argument('--document', help='Select a registered scientific document for evidence checks')
    parser.add_argument('--allow-pending', action='store_true', help='Report unresolved evidence explicitly without failing only for pending bindings')
    args = parser.parse_args()
    if args.command != 'evidence' and (args.document or args.allow_pending):
        parser.error('--document and --allow-pending apply only to evidence')
    if args.command == 'evidence':
        from .evidence import check
        report = check(document=args.document, strict=not args.allow_pending)
        print(json.dumps(report.result(), indent=2))
        raise SystemExit(0 if report.ok() else 1)
    if args.command == 'verify':
        from .verify import verify
        result = verify()
    elif args.command == 'tables':
        from .tables import reproduce
        result = reproduce(args.output)
    else:
        from .figures import reproduce
        result = reproduce(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
