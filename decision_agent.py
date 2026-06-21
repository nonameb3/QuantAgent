"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""

import json
import re


def _parse_llm_json(raw: str) -> dict:
    """
    Extract and parse a JSON object from LLM output.
    Handles markdown fences (```json ... ```) and trailing commas.
    Raises ValueError if no valid JSON object is found.
    """
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    candidate = fenced.group(1).strip() if fenced else raw.strip()

    # Remove trailing commas before } or ] (common LLM mistake)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # Try parsing the cleaned candidate first
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back: find the first {...} block in the original text
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        fallback = re.sub(r",\s*([}\]])", r"\1", brace_match.group(0))
        try:
            return json.loads(fallback)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in LLM output: {raw[:200]!r}")


def _validate_decision(data: dict) -> dict:
    """
    Validate required fields and normalise values.
    Returns the validated dict or raises ValueError.
    """
    required = {"decision", "justification", "risk_reward_ratio", "forecast_horizon"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing required fields in decision JSON: {missing}")

    decision = str(data["decision"]).upper().strip()
    if decision not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid decision value: {data['decision']!r}. Must be LONG or SHORT.")
    data["decision"] = decision

    try:
        rr = float(data["risk_reward_ratio"])
    except (TypeError, ValueError):
        raise ValueError(f"risk_reward_ratio is not numeric: {data['risk_reward_ratio']!r}")
    if not (1.0 <= rr <= 3.0):
        raise ValueError(f"risk_reward_ratio {rr} is outside expected range [1.0, 3.0]")
    data["risk_reward_ratio"] = rr

    return data


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        # Gate: skip if any upstream agent failed
        if not state.get("signal_valid", True):
            errors = state.get("agent_errors", {})
            return {
                "final_trade_decision": json.dumps({
                    "decision": "SKIPPED",
                    "justification": f"Signal invalidated by upstream agent errors: {errors}",
                    "risk_reward_ratio": None,
                    "forecast_horizon": None,
                }),
                "messages": [],
                "decision_prompt": "",
            }

        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # --- System prompt for LLM ---
        prompt = f"""You are a high-frequency quantitative trading (HFT) analyst operating on the current {time_frame} K-line chart for {stock_name}. Your task is to issue an **immediate execution order**: **LONG** or **SHORT**. ⚠️ HOLD is prohibited due to HFT constraints.

            Your decision should forecast the market move over the **next N candlesticks**, where:
            - For example: TIME_FRAME = 15min, N = 1 → Predict the next 15 minutes.
            - TIME_FRAME = 4hour, N = 1 → Predict the next 4 hours.

            Base your decision on the combined strength, alignment, and timing of the following three reports:

            ---

            ### 1. Technical Indicator Report:
            - Evaluate momentum (e.g., MACD, ROC) and oscillators (e.g., RSI, Stochastic, Williams %R).
            - Give **higher weight to strong directional signals** such as MACD crossovers, RSI divergence, extreme overbought/oversold levels.
            - **Ignore or down-weight neutral or mixed signals** unless they align across multiple indicators.

            ---

            ### 2. Pattern Report:
            - Only act on bullish or bearish patterns if:
            - The pattern is **clearly recognizable and mostly complete**, and
            - A **breakout or breakdown is already underway** or highly probable based on price and momentum (e.g., strong wick, volume spike, engulfing candle).
            - **Do NOT act** on early-stage or speculative patterns. Do not treat consolidating setups as tradable unless there is **breakout confirmation** from other reports.

            ---

            ### 3. Trend Report:
            - Analyze how price interacts with support and resistance:
            - An **upward sloping support line** suggests buying interest.
            - A **downward sloping resistance line** suggests selling pressure.
            - If price is compressing between trendlines:
            - Predict breakout **only when confluence exists with strong candles or indicator confirmation**.
            - **Do NOT assume breakout direction** from geometry alone.

            ---

            ### ✅ Decision Strategy

            1. Only act on **confirmed** signals — avoid emerging, speculative, or conflicting signals.
            2. Prioritize decisions where **all three reports** (Indicator, Pattern, and Trend) **align in the same direction**.
            3. Give more weight to:
            - Recent strong momentum (e.g., MACD crossover, RSI breakout)
            - Decisive price action (e.g., breakout candle, rejection wicks, support bounce)
            4. If reports disagree:
            - Choose the direction with **stronger and more recent confirmation**
            - Prefer **momentum-backed signals** over weak oscillator hints.
            5. ⚖️ If the market is in consolidation or reports are mixed:
            - Default to the **dominant trendline slope** (e.g., SHORT in descending channel).
            - Do not guess direction — choose the **more defensible** side.
            6. Suggest a reasonable **risk-reward ratio** between **1.2 and 1.8**, based on current volatility and trend strength.

            ---
            ### 🧠 Output Format in json(for system parsing):

            ```
            {{
            "forecast_horizon": "Predicting next 3 candlestick (15 minutes, 1 hour, etc.)",
            "decision": "<LONG or SHORT>",
            "justification": "<Concise, confirmed reasoning based on reports>",
            "risk_reward_ratio": "<float between 1.2 and 1.8>",
            }}

            --------
            **Technical Indicator Report**
            {indicator_report}

            **Pattern Report**
            {pattern_report}

            **Trend Report**
            {trend_report}

        """

        # --- LLM call for decision ---
        response = llm.invoke(prompt)
        raw_content = response.content

        # --- Parse and validate JSON output ---
        agent_errors = dict(state.get("agent_errors") or {})
        confidence_scores = dict(state.get("confidence_scores") or {})

        try:
            parsed = _parse_llm_json(raw_content)
            validated = _validate_decision(parsed)
            final_decision = json.dumps(validated)
            confidence_scores["decision"] = 1.0
        except (ValueError, KeyError) as exc:
            agent_errors["decision"] = str(exc)
            final_decision = json.dumps({
                "decision": "ERROR",
                "justification": f"Failed to parse LLM output: {exc}",
                "raw_output": raw_content[:500],
                "risk_reward_ratio": None,
                "forecast_horizon": None,
            })
            confidence_scores["decision"] = 0.0

        return {
            "final_trade_decision": final_decision,
            "agent_errors": agent_errors,
            "confidence_scores": confidence_scores,
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node
