"""Focus API review text on changed arguments without changing scan evidence."""

import ast
import difflib
import re


BINDING = re.compile(
    r"Parameter `([^`]+)` changed its accepted calling convention or positional binding\.\s*"
)


def signature(value):
    declaration, _, aliases = str(value or "").partition("  # ")
    try:
        node = ast.parse(declaration + ":\n pass").body[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return declaration.split("(", 1)[0], node, aliases
    except (SyntaxError, ValueError):
        return None


def annotations(node):
    args = node.args
    return {
        a.arg: ast.unparse(a.annotation) if a.annotation else ""
        for a in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ]
        if a is not None
    }


def parameters(node, other):
    peers = annotations(other)

    def argument(arg, default=None):
        result = arg.arg
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        if annotation and annotation != peers.get(arg.arg):
            result += ": " + annotation
        if default is not None:
            result += "=" + ast.unparse(default)
        return result

    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    result = []
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        result.append(argument(arg, default))
        if args.posonlyargs and index + 1 == len(args.posonlyargs):
            result.append("/")
    if args.vararg:
        result.append("*" + argument(args.vararg))
    elif args.kwonlyargs:
        result.append("*")
    result.extend(
        argument(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    if args.kwarg:
        result.append("**" + argument(args.kwarg))
    return result


def focused_pair(before, after):
    old, new = signature(before), signature(after)
    if not old or not new:
        pair = [str(before or ""), str(after or "")]
        for index, parsed in enumerate((old, new)):
            if parsed and len(pair[index]) > 220:
                pair[index] = parsed[0] + "(...)"
        if pair != [str(before or ""), str(after or "")]:
            return pair
        return None
    left, right = parameters(old[1], new[1]), parameters(new[1], old[1])
    output = [[], []]
    operations = difflib.SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes()
    for index, (kind, a, b, c, d) in enumerate(operations):
        if kind != "equal":
            output[0].extend(left[a:b])
            output[1].extend(right[c:d])
            continue
        same = left[a:b]
        if len(same) > 2:
            same = (
                ([same[0]] if index else [])
                + ["..."]
                + ([same[-1]] if index + 1 < len(operations) else [])
            )
        for side in output:
            side.extend(same)
    old_return = ast.unparse(old[1].returns) if old[1].returns else ""
    new_return = ast.unparse(new[1].returns) if new[1].returns else ""
    result = []
    for item, args, returns in zip(
        (old, new), output, (old_return, new_return), strict=True
    ):
        value = item[0] + "(" + ", ".join(args) + ")"
        if returns and old_return != new_return:
            value += " -> " + returns
        if item[2] and old[2] != new[2]:
            value += "  # " + item[2]
        result.append(value)
    return result


def explanation(finding):
    message = str(finding.get("message") or "")
    names = list(dict.fromkeys(BINDING.findall(message)))
    if not names:
        return message
    remaining = BINDING.sub("", message).strip()
    old, new = signature(finding.get("before")), signature(finding.get("after"))
    focus = "Positional or keyword argument binding changed. Review existing calls."
    if old and new:
        before = [a.arg for a in [*old[1].args.posonlyargs, *old[1].args.args]]
        after = [a.arg for a in [*new[1].args.posonlyargs, *new[1].args.args]]
        equal = {
            name
            for kind, a, b, _, _ in difflib.SequenceMatcher(
                a=before, b=after, autojunk=False
            ).get_opcodes()
            if kind == "equal"
            for name in before[a:b]
        }
        moved = [n for n in before if n in after and n not in equal]
        optional = (
            set(after[-len(new[1].args.defaults) :]) if new[1].args.defaults else set()
        )
        old_optional = (
            set(before[-len(old[1].args.defaults) :]) if old[1].args.defaults else set()
        )
        optional_moves = [
            name
            for name in before
            if name in optional - old_optional
            and name in after
            and before.index(name) != after.index(name)
        ]
        if len(optional_moves) == 1:
            moved = optional_moves
        if len(moved) == 1:
            name = moved[0]
            position = (
                "to the end"
                if after[-1] == name
                else "to the beginning"
                if after[0] == name
                else "to a different position"
            )
            extra = (
                " and became optional"
                if name in optional and name not in old_optional
                else ""
            )
            focus = f"`{name}` moved {position}{extra}. Existing positional calls may bind arguments incorrectly."
        elif moved:
            focus = "Positional parameters were reordered. Existing positional calls may bind arguments incorrectly."
        elif added := [n for n in after if n not in before]:
            labels = ", ".join(f"`{n}`" for n in added)
            focus = f"New positional parameters ({labels}) shift existing argument bindings."
        elif keyword_only := [a.arg for a in new[1].args.kwonlyargs if a.arg in names]:
            labels = ", ".join(f"`{n}`" for n in keyword_only)
            focus = f"Parameters {labels} became keyword-only. Old positional calls passing them fail."
    return " ".join(part for part in (focus, remaining) if part)
