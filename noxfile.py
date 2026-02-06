import nox

PYTHONS = ["3.10", "3.11", "3.12"]


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".")
    session.run("black", "--check", ".")


@nox.session
def format(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".", "--fix")
    session.run("black", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("mypy", "src")
