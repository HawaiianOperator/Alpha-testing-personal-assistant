# Explicit keyword commands

These are deterministic command rules alongside the trained visual model. They do not require an AI model to guess a known keyword.

| Input | Result |
| --- | --- |
| Google Nvidia | Prepare https://www.google.com/search?q=Nvidia |
| Search Nvidia | Prepare the same Google query |
| Compute 2 + 2 | Return 4 locally |
| Google | Ask what to search for |
| Compute | Ask what to calculate |
| comp | Suggest the keyword completion compute |

Keywords are case-insensitive. A colon is optional, such as Search: Nvidia. Nvidia is an example query, not a hardcoded destination for every Google command.

## Try the commands

    python -m local_model.commands "Google Nvidia"
    python -m local_model.commands "Search Nvidia"
    python -m local_model.commands "Compute 2 + 2"
    python -m local_model.commands --suggest comp

By default, search routing returns the URL without contacting Google. Add --open explicitly on a machine with a browser to request opening it:

    python -m local_model.commands "Google Nvidia" --open

That action sends the selected query to Google. Calculations remain local.

## Arithmetic

Numbers, decimals, parentheses, and +, -, *, / are supported. The evaluator walks a restricted syntax tree and uses Decimal; it never executes eval, exec, Python names, function calls, file access, or shell commands.

Request length, expression length, tree size/depth, numeric exponents, and supported magnitude are bounded. Invalid input or division by zero yields a clarification response.

## Clarification and autocomplete

KeywordSession stores only the current pending command label. For example, Google then Nvidia forms one current search request. Completed tasks and cancellation clear this state.

Autocomplete completes the known keyword prefix from the current input. It is not yet a general-language completion model and does not learn from task history.

Caller-side UI/queue code must pass results to the appropriate executor and clear its own private input/output buffers. The CLI deliberately prints a response; terminal history/output storage is controlled by the user's shell and is outside the RAM-only active-service guarantee.

## Remaining product scope

The desired broader system may accept user-defined tasks involving research or calendar optimization. Those require tool integrations, planning, data, and evaluation beyond this keyword layer. Unknown commands currently ask for a supported command rather than pretending to understand them.

## Capability boundary

Training from scratch remains mandatory. Unsupported commands, requests over the input limit, and unsupported calculations return status `rejected` with exactly: "I am sorry the task is too complicated, try an less complicated task". Empty requests and missing command arguments ask for clarification. Rejection clears the pending command and does not execute a partial task.

This boundary is deterministic, not a trained general difficulty estimator. Search only prepares or opens the literal query; it does not compare results or complete actions described inside the query. The trained visual classifier is separate and does not understand arbitrary tasks.
