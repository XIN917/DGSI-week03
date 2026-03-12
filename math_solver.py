#!/usr/bin/env python3
"""
Math Solver CLI
===============
High-school math assistant powered by an OpenAI-compatible chat completions
API with function calling.

Available tools
---------------
- evaluate_expression  – numerically/symbolically evaluate any expression
- solve_equation       – solve an equation for a variable
- factor_expression    – factor a polynomial / algebraic expression
- plot_function        – plot f(x) over a range and save a PNG

Configuration (.env in the same directory)
------------------------------------------
OPENAI_API_KEY      – required
OPENAI_API_ENDPOINT – optional (e.g. for Qwen / Azure / local models)
MODEL               – optional (default: gpt-4o)

Usage
-----
    python math_solver.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

console = Console()

# Plots are stored relative to this file so the CLI works from any cwd.
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── Math tool implementations ──────────────────────────────────────────────────

def evaluate_expression(expression: str) -> str:
    """Evaluate or simplify a mathematical expression."""
    try:
        expr = sp.sympify(expression, evaluate=True)
        simplified = sp.simplify(expr)
        exact_str = str(simplified)
        if simplified.is_number:
            if simplified.is_integer:
                return exact_str
            num_val = float(sp.N(simplified))
            num_str = f"{num_val:.6g}"
            # Avoid "0.5  ≈  0.5"-style duplicates
            if exact_str == num_str:
                return exact_str
            return f"{exact_str}  ≈  {num_str}"
        return exact_str
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve an equation for the specified variable and return all solutions."""
    try:
        var = sp.Symbol(variable)
        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            expr = sp.sympify(lhs_str.strip()) - sp.sympify(rhs_str.strip())
        else:
            expr = sp.sympify(equation)
        solutions = sp.solve(expr, var)
        if not solutions:
            return "No solutions found."
        return f"{variable} = " + ",  ".join(str(s) for s in solutions)
    except Exception as exc:
        return f"Error solving '{equation}': {exc}"


def factor_expression(expression: str) -> str:
    """Factor a polynomial or algebraic expression."""
    try:
        expr = sp.sympify(expression)
        factored = sp.factor(expr)
        return str(factored)
    except Exception as exc:
        return f"Error factoring '{expression}': {exc}"


def plot_function(
    expression: str,
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> str:
    """
    Plot f(x) = *expression* over [x_min, x_max] and save a PNG to the plots
    folder.  Returns the absolute path to the saved file.
    """
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(expression)
        f_lambdified = sp.lambdify(x, expr, modules=["numpy"])

        xs = np.linspace(float(x_min), float(x_max), 800)
        ys = np.asarray(f_lambdified(xs), dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, linewidth=2, color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"f(x) = {expression}", fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        # Build a filesystem-safe name and make it unique
        safe = "".join(c if c.isalnum() or c in "._+-" else "_" for c in expression)
        path = PLOTS_DIR / f"{safe}.png"
        counter = 1
        while path.exists():
            path = PLOTS_DIR / f"{safe}_{counter}.png"
            counter += 1

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    except Exception as exc:
        return f"Error plotting '{expression}': {exc}"


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict = {
    "evaluate_expression": evaluate_expression,
    "solve_equation": solve_equation,
    "factor_expression": factor_expression,
    "plot_function": plot_function,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": (
                "Evaluate or simplify a mathematical expression and return its exact "
                "or numerical value. Use for arithmetic, algebra, trigonometry, "
                "logarithms, or any expression that does not need solving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A SymPy-compatible expression string. "
                            "Use ** for powers, sqrt(), sin(), cos(), tan(), "
                            "log(), exp(), pi, E, etc. "
                            "Examples: '2**10 + sqrt(16)', 'sin(pi/6)', 'log(E**3)'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": (
                "Solve an algebraic equation for a given variable and return all "
                "solutions (real and complex). Supply the equation with '=' to "
                "separate left-hand and right-hand sides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": (
                            "The equation to solve. Use SymPy syntax. "
                            "Examples: 'x**2 - 5*x + 6 = 0', '2*x + 3 = 11'."
                        ),
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to solve for (default: 'x').",
                    },
                },
                "required": ["equation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "factor_expression",
            "description": (
                "Factor a polynomial or algebraic expression into its irreducible "
                "factors over the integers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The expression to factor. Use SymPy syntax. "
                            "Examples: 'x**2 - 5*x + 6', 'x**3 - 1'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_function",
            "description": (
                "Plot a function of x over a specified range and save it as a PNG "
                "file. Returns the absolute path to the saved image. "
                "Use this whenever the user asks for a graph or a plot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The function of x to plot. Use SymPy/NumPy syntax. "
                            "Examples: 'sin(x)', 'x**3 - 3*x + 2', 'exp(-x**2)'."
                        ),
                    },
                    "x_min": {
                        "type": "number",
                        "description": "Lower bound of the x-axis (default: -10).",
                    },
                    "x_max": {
                        "type": "number",
                        "description": "Upper bound of the x-axis (default: 10).",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ── Agent loop ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful high-school math tutor. "
    "When the user gives you a math problem use the available tools to solve it "
    "step by step, calling as many tools as needed. "
    "Always explain your reasoning in plain English before and after using a tool. "
    "Use SymPy-compatible syntax for all expressions passed to tools "
    "(** for powers, sqrt(), sin(), cos(), log(), pi, E, etc.). "
    "When you plot a function, tell the user where the image was saved."
)


def _dispatch_tool(tool_call) -> str:
    """Execute a single tool call and return its string result."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return f"Error parsing arguments: {exc}"

    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"

    label = ", ".join(f"{k}={v!r}" for k, v in args.items())
    console.print(f"  [dim cyan]→ {name}({label})[/dim cyan]")
    result = fn(**args)
    console.print(f"  [dim green]← {result}[/dim green]")
    return str(result)


def run_agent(client: OpenAI, model: str, user_problem: str) -> None:
    """Run the agentic loop for a single user problem."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_problem},
    ]

    console.print()

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # Build the assistant message dict manually so it round-trips cleanly
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # No tool calls → final answer
        if not msg.tool_calls:
            content = msg.content or ""
            console.print(
                Panel(
                    Markdown(content),
                    title="[bold green]Answer[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            break

        # Execute every requested tool and feed results back
        for tc in msg.tool_calls:
            result = _dispatch_tool(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_ENDPOINT") or None
    model = os.getenv("MODEL", "gpt-4o")

    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY is not set. "
            "Add it to a .env file in the same directory as this script."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    console.print(
        Panel(
            "[bold cyan]Math Solver[/bold cyan]  —  AI tutor powered by function calling\n"
            "[dim]Tools: evaluate · solve · factor · plot[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print(
        "[dim]Type your math problem and press Enter.  "
        "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.[/dim]\n"
    )

    while True:
        try:
            problem = console.input("[bold yellow]Problem:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not problem:
            continue
        if problem.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Bye![/dim]")
            break

        run_agent(client, model, problem)
        console.print()


if __name__ == "__main__":
    main()
