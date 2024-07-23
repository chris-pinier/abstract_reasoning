import argparse
from rich.console import Console
from rich.markdown import Markdown

console = Console()


# Custom Help Formatter using Rich
class RichHelpFormatter(argparse.HelpFormatter):
    def _format_action(self, action):
        text = super()._format_action(action)
        return Markdown(f"**{text}**")

    def _format_usage(self, usage, actions, groups, prefix):
        usage_text = super()._format_usage(usage, actions, groups, prefix)
        return Markdown(f"`{usage_text}`")


# Custom ArgumentParser to use Rich for error messages
class RichArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        console.print("[bold red]Error:[/bold red]", message)
        self.print_help()
        self.exit(2)


def main():
    parser = RichArgumentParser(
        description="An example application using Rich with Argparse.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )

    args = parser.parse_args()

    if args.verbose:
        console.print("[bold green]Verbose mode is on.[/bold green]")
    else:
        console.print("[bold yellow]Verbose mode is off.[/bold yellow]")


if __name__ == "__main__":
    main()
