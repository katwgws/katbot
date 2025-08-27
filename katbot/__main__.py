# ⋅⋆•°☙ katbot/__main__.py ❧°•⋆⋅

import logging

import click

from .app import generate, post_one

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-g",
    "--generate",
    "generate_count",
    type=click.IntRange(0, None),
    default=0,
    help="Generate N tweets and save without posting.",
)
def cli(generate_count: int) -> None:
    if generate_count and generate_count > 0:
        generate(generate_count)
    else:
        post_one()


cli()
