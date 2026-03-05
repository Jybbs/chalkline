import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 📐 Chalkline

        **Career mapping for Maine's construction industry.**

        Upload a resume to see where you sit in Maine's construction
        landscape, what skills separate you from your next role, and
        how to get there.
        """
    )
    return


if __name__ == "__main__":
    app.run()
