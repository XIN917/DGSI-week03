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
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

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


# ── UI helpers (matching three_pigs_function_calling.py style) ──────────────────

def create_message_panel(role: str, content: str) -> Panel:
    """Create a styled panel for a chat message."""
    styles = {
        "user":      ("bright_white on blue",      "blue",    "🧠 You"),
        "assistant": ("bright_white on dark_green", "green",   "🤖 Assistant"),
        "system":    ("bright_white on purple4",    "magenta", "⚙️ System"),
        "tool":      ("black on yellow",            "yellow",  "🔧 Tool Result"),
    }
    text_style, border_color, title = styles.get(role, ("bright_white on grey23", "white", role))
    return Panel(
        Text(content, style=text_style),
        title=title,
        title_align="left",
        border_style=border_color,
        padding=(0, 1),
    )


def show_context_stack(messages: list) -> Panel:
    """Show the current conversation context as a table."""
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_white on grey23",
        style="on grey23",
    )
    table.add_column("#",    style="bright_cyan on grey23",    width=3)
    table.add_column("Role", style="bright_magenta on grey23", width=12)
    table.add_column("Content Preview", style="bright_white on grey23")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        preview = content.replace("\n", " ") if content else "(tool_calls)"
        table.add_row(str(i), role, preview[:120])
    tool_names = ", ".join(TOOL_FUNCTIONS.keys())
    return Panel(
        table,
        title=f"📚 Context Stack ({len(messages)} messages) | Tools: [{tool_names}]",
        border_style="magenta",
        style="on grey23",
        padding=(0, 1),
    )


def show_api_request(request_data: dict) -> Panel:
    """Show the outgoing API request as syntax-highlighted JSON."""
    json_str = json.dumps(request_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(
        syntax,
        title="📤 API Request (sent to model)",
        border_style="yellow",
        style="on grey23",
        padding=(0, 1),
    )


def show_api_response(response_data: dict) -> Panel:
    """Show the incoming API response as syntax-highlighted JSON."""
    json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(
        syntax,
        title="📥 API Response (from model)",
        border_style="cyan",
        style="on grey23",
        padding=(0, 1),
    )


def wait_for_llm():
    """Live spinner displayed while waiting for the model."""
    return Live(
        Panel(
            Spinner("dots", text=Text(" Waiting for model response...", style="bold black on yellow")),
            border_style="yellow",
            style="on yellow",
            padding=(0, 1),
        ),
        console=console,
        refresh_per_second=10,
    )


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


def _dispatch_tool(tool_call, messages: list) -> str:
    """Execute a single tool call, display Rich panels, append result to messages."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        err = f"Error parsing arguments: {exc}"
        console.print(create_message_panel("tool", err))
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": err})
        return err

    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        err = f"Unknown tool: {name}"
        console.print(create_message_panel("tool", err))
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": err})
        return err

    # Show the function call (yellow panel like three_pigs)
    args_display = ",\n   ".join(f'{k}={v!r}' for k, v in args.items())
    console.print()
    console.print(
        Panel(
            Text(f"🔧 FUNCTION CALLED: {name}(\n   {args_display}\n)", style="bold black on yellow"),
            border_style="yellow",
            style="on yellow",
            padding=(0, 1),
        )
    )

    result = fn(**args)

    # Show the tool result
    console.print()
    console.print(create_message_panel("tool", str(result)))

    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
    return str(result)


def run_agent(client: OpenAI, model: str, user_problem: str) -> None:
    """Run the agentic loop for a single user problem."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_problem},
    ]

    # Show user message and initial context
    console.print()
    console.print(create_message_panel("user", user_problem))
    console.print()
    console.print(show_context_stack(messages))

    while True:
        # Build request payload for display
        request_data = {
            "model":    model,
            "messages": messages,
            "tools":    TOOLS,
            "tool_choice": "auto",
        }
        console.print()
        console.print(show_api_request(request_data))

        # Call the API with a spinner
        with wait_for_llm():
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

        msg = response.choices[0].message

        # Show the raw response
        response_data: dict = {
            "id":            response.id,
            "model":         response.model,
            "finish_reason": response.choices[0].finish_reason,
            "message": {
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": None,
            },
        }
        if msg.tool_calls:
            response_data["message"]["tool_calls"] = [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        console.print()
        console.print(show_api_response(response_data))

        # Build the assistant entry
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # If the assistant wrote text alongside tool_calls, show it
        if msg.content and msg.tool_calls:
            console.print()
            console.print(create_message_panel("assistant", msg.content))

        # No tool calls → final answer
        if not msg.tool_calls:
            content = msg.content or ""
            console.print()
            console.print(
                Panel(
                    Markdown(content),
                    title="[bold green]✅ Answer[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()
            console.print(show_context_stack(messages))
            break

        # Execute every requested tool
        for tc in msg.tool_calls:
            _dispatch_tool(tc, messages)

        # Show updated context before next API round
        console.print()
        console.print(show_context_stack(messages))


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_ENDPOINT") or None
    model    = os.getenv("MODEL", "gpt-4o")

    if not api_key:
        console.print(
            Panel(
                Text(
                    "OPENAI_API_KEY not found!\n\n"
                    "Create a .env file in the same directory with:\n"
                    "OPENAI_API_KEY=your-key-here",
                    style="bold bright_white on dark_red",
                ),
                title="❌ Error",
                border_style="red",
                style="on dark_red",
            )
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    console.clear()

    # Welcome panel
    console.print(
        Panel(
            Text.from_markup(
                "[bold bright_white]🧠 Math Solver — AI Tutor[/bold bright_white]\n\n"
                "[bright_cyan]Powered by function calling[/bright_cyan]\n\n"
                "[bright_white]Type a high-school math problem in natural language.\n"
                "The model will use tools to solve it step by step.[/bright_white]"
            ),
            title="📚 Math Solver 📚",
            title_align="center",
            border_style="bright_magenta",
            style="on grey23",
            padding=(1, 4),
        )
    )

    # Config panel
    console.print()
    console.print(
        Panel(
            Text(
                f"Model:    {model}\n"
                f"Endpoint: {base_url or 'https://api.openai.com/v1'}\n"
                f"Tools:    evaluate_expression · solve_equation · factor_expression · plot_function",
                style="bright_white on grey23",
            ),
            title="⚙️ Configuration",
            border_style="cyan",
            style="on grey23",
        )
    )

    console.print()
    console.print("[dim]Type your math problem and press Enter.  Type [bold]quit[/bold] or [bold]exit[/bold] to stop.[/dim]\n")

    while True:
        try:
            problem = console.input("[bold yellow]🧠 Problem:[/bold yellow] ").strip()
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
