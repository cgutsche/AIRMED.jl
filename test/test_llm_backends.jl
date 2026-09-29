"""
LLM backend and response-parsing tests. The HTTP backends are exercised
against a scripted mock server on 127.0.0.1, so no request leaves the machine
and no API key is needed. The Anthropic backends have a fixed endpoint and are
covered only on their no-key path.

Relies on `problem` and the constants from electrical_fixture.jl, which
test_electrical.jl includes once — see the ordering note in runtests.jl.
"""

using Test
using HTTP
using JSON3

# Serve `replies` in order (the last one repeats) on a loopback port and pass
# the `/v1` base URL plus the list of requested paths to `f`. Sockets is taken
# from HTTP, since the stdlib is not a declared test dependency.
function with_mock_llm(f, replies::Vector{String}; status::Int = 200)
    queue    = copy(replies)
    requests = String[]
    port, sock = HTTP.Sockets.listenany(HTTP.Sockets.IPv4("127.0.0.1"), 18000)
    srv = HTTP.serve!("127.0.0.1", port; server = sock) do req
        push!(requests, req.target)
        body = length(queue) > 1 ? popfirst!(queue) : queue[1]
        HTTP.Response(status, ["content-type" => "application/json"], body)
    end
    try
        return f("http://127.0.0.1:$port/v1", requests)
    finally
        close(srv)
    end
end

openai_reply(content; finish = "stop") = JSON3.write(Dict(
    "choices" => [Dict("message" => Dict("content" => content, "finish_reason" => finish))]))
ollama_reply(content; done_reason = "stop", thinking = "") = JSON3.write(Dict(
    "message" => Dict("content" => content, "thinking" => thinking), "done_reason" => done_reason))

proposal_json(ps...) = "```json\n" * JSON3.write(Dict("proposals" => collect(ps))) * "\n```"
by_hook(hook, type; name = "extra") =
    Dict("hook" => hook, "name" => name, "component_type" => type, "rationale" => "test")

const HOOKS = problem.component_hooks

@testset "LLM backends and parsing" begin

    @testset "_parse_llm_proposals: hook-name schema resolves ports from the hook" begin
        props, code = AIRMED._parse_llm_proposals(
            proposal_json(by_hook("r1_to_cap", "Resistor")), HOOKS)
        @test length(props) == 1
        @test props[1].component_type == :Resistor
        @test props[1].port_a == (:r1, :n) && props[1].port_b == (:cap, :p)
        @test code == ""
    end

    @testset "_parse_llm_proposals: legacy port schema with caller aliases" begin
        txt = """{"proposals": [{"name": "rx", "component_type": "Resistor",
                  "port_a": ["r1", "+"], "port_b": ["r1", "-"]}],
                  "julia_model_code": "x = 1"}"""
        props, code = AIRMED._parse_llm_proposals(txt, HOOKS;
                                                  aliases = Dict("+" => "p", "-" => "n"))
        @test props[1].port_a == (:r1, :p) && props[1].port_b == (:r1, :n)
        @test code == "x = 1"
    end

    @testset "_parse_llm_proposals: defaults and rejected entries" begin
        # Missing component_type falls back to the caller's default, else :Unknown.
        no_type = proposal_json(Dict("hook" => "r1_parallel"))
        @test AIRMED._parse_llm_proposals(no_type, HOOKS;
                  default_component_type = :Capacitor)[1][1].component_type == :Capacitor
        @test AIRMED._parse_llm_proposals(no_type, HOOKS)[1][1].component_type == :Unknown

        bad_hook = proposal_json(by_hook("no_such_hook", "Resistor"))
        props = @test_logs (:warn, r"hook \"no_such_hook\" not found") AIRMED._parse_llm_proposals(bad_hook, HOOKS)
        @test isempty(props[1])

        short_port = """{"proposals": [{"name": "rx", "port_a": ["r1"], "port_b": ["r1", "n"]}]}"""
        props = @test_logs (:warn, r"port_a needs") AIRMED._parse_llm_proposals(short_port, HOOKS)
        @test isempty(props[1])
    end

    @testset "_parse_llm_proposals: unusable responses" begin
        @test (@test_logs (:warn, r"no parseable JSON") AIRMED._parse_llm_proposals("no json here")) ==
              (ComponentProposal[], "")
        @test (@test_logs (:warn, r"Failed to parse") AIRMED._parse_llm_proposals("""{"proposals": [}""")) ==
              (ComponentProposal[], "")
    end

    @testset "JSON extraction repairs what models commonly emit" begin
        # Comments, including a `//` that sits inside a string and must survive.
        commented = """{ // a comment
          "proposals": [], # another
          "julia_model_code": "see http://example.org \\"quoted\\" µ"
        }"""
        obj = JSON3.read(AIRMED._extract_json_block(commented))
        @test obj["julia_model_code"] == "see http://example.org \"quoted\" µ"

        # A truncated response gets its missing closing braces appended.
        @test AIRMED._brace_extract("""{"a": {"b": 1""") == """{"a": {"b": 1}}"""
        @test isnothing(AIRMED._brace_extract("no braces"))

        # A fence without an object: fall back to searching the whole text.
        fenced = "```\nnothing here\n```\n{\"proposals\": []}"
        @test AIRMED._extract_json_block(fenced) == "{\"proposals\": []}"

        # Triple-quoted and backtick strings are not valid JSON; both are blanked.
        tq = "{\"julia_model_code\": \"\"\"x = 1\"\"\", \"proposals\": []}"
        @test JSON3.read(AIRMED._extract_json_block(tq))["julia_model_code"] == ""
        bt = "{\"julia_model_code\": `x = 1`, \"proposals\": []}"
        @test JSON3.read(AIRMED._extract_json_block(bt))["julia_model_code"] == ""
    end

    @testset "_call_llm: OpenAI-compatible backend" begin
        with_mock_llm([openai_reply(proposal_json(by_hook("r1_parallel", "Resistor")); finish = "length")]) do url, reqs
            logf = tempname()
            text, props, _ = @test_logs (:warn, r"truncated") match_mode = :any AIRMED._call_llm(
                :openai, "the prompt", ""; base_url = url, hooks = HOOKS, log_file = logf)
            @test only(reqs) == "/v1/chat/completions"
            @test occursin("proposals", text)
            @test length(props) == 1 && props[1].port_a == (:r1, :p)

            log = read(logf, String)
            @test occursin("AIRMED LLM CALL", log)
            @test occursin("Proposals parsed: 1", log)
            @test occursin("--- PROMPT ---\nthe prompt", log)
            rm(logf)
        end
    end

    @testset "_call_llm: Ollama native backend" begin
        with_mock_llm([ollama_reply(proposal_json(by_hook("r1_to_cap", "Resistor")); done_reason = "length")]) do url, reqs
            _, props, _ = @test_logs (:warn, r"num_predict") match_mode = :any AIRMED._call_llm(
                :ollama_native, "p", ""; base_url = url, hooks = HOOKS, num_ctx = 2048, think = false)
            @test only(reqs) == "/api/chat"          # the /v1 suffix is stripped
            @test length(props) == 1
        end
        with_mock_llm([ollama_reply(""; thinking = "long reasoning")]) do url, _
            _, props, _ = @test_logs (:warn, r"empty content") match_mode = :any AIRMED._call_llm(
                :ollama_native, "p", ""; base_url = url, hooks = HOOKS)
            @test isempty(props)
        end
    end

    @testset "_call_llm: failures and stubs never throw" begin
        with_mock_llm(["server error"]; status = 500) do url, _
            text, props, _ = AIRMED._call_llm(:openai, "p", ""; base_url = url)
            @test startswith(text, "[openai error") && isempty(props)
            text, _, _ = AIRMED._call_llm(:ollama_native, "p", ""; base_url = url)
            @test startswith(text, "[ollama error")
        end
        @test startswith(first(AIRMED._call_llm(:openai, "p", "")), "[openai stub")
        @test startswith(first(AIRMED._call_llm(:anthropic, "p", "")), "[anthropic stub")
        @test startswith(first(AIRMED._call_llm(:no_such_api, "p", "")), "[unknown api")
        @test startswith(AIRMED._call_llm_text(:none, "p", ""), "[no LLM configured")

        with_mock_llm([openai_reply("plain answer")]) do url, _
            @test AIRMED._call_llm_text(:openai, "p", ""; base_url = url) == "plain answer"
        end
    end

    @testset "_call_claude_cli against stand-in executables" begin
        # Tiny scripts replace the real `claude` binary, so no model is invoked.
        mktempdir() do dir
            function fake(name, win, unix)
                path = joinpath(dir, Sys.iswindows() ? "$name.bat" : "$name.sh")
                write(path, Sys.iswindows() ? win : unix)
                Sys.iswindows() || chmod(path, 0o755)
                return path
            end
            json = """{"proposals": [{"hook": "r1_parallel", "component_type": "Resistor"}]}"""
            ok   = fake("ok",   "@echo off\r\nmore > nul\r\necho $json\r\n",
                                "#!/bin/sh\ncat > /dev/null\necho '$json'\n")
            fail = fake("fail", "@echo off\r\necho boom 1>&2\r\nexit /b 3\r\n",
                                "#!/bin/sh\necho boom 1>&2\nexit 3\n")
            slow = fake("slow", "@echo off\r\nping -n 4 127.0.0.1 > nul\r\n",
                                "#!/bin/sh\nsleep 3\n")

            _, props, _ = AIRMED._call_claude_cli("p"; cli_path = ok, hooks = HOOKS, workdir = dir)
            @test length(props) == 1 && props[1].port_a == (:r1, :p)
            # `base_url` carries the executable path through the dispatcher.
            _, props, _ = AIRMED._call_llm(:claude_cli, "p", ""; base_url = ok, hooks = HOOKS)
            @test length(props) == 1

            text, props, _ = AIRMED._call_claude_cli("p"; cli_path = fail)
            @test startswith(text, "[claude_cli error: exit 3") && occursin("boom", text)
            @test isempty(props)

            text, _, _ = AIRMED._call_claude_cli("p"; cli_path = slow, timeout = 1)
            @test startswith(text, "[claude_cli error: timeout")

            text, _, _ = @test_logs (:warn, r"Claude CLI call failed") AIRMED._call_claude_cli(
                "p"; cli_path = joinpath(dir, "does_not_exist"))
            @test startswith(text, "[claude_cli error:")
        end
    end

    @testset "_write_llm_log: nothing and unwritable paths" begin
        @test isnothing(AIRMED._write_llm_log(nothing, :openai, "", "", "p", "r", ComponentProposal[]))
        mktempdir() do dir   # a directory cannot be opened for appending
            @test_logs (:warn, r"LLM log write failed") AIRMED._write_llm_log(
                dir, :openai, "m", "u", "p", "r", ComponentProposal[])
        end
    end

    @testset "_try_build_model" begin
        m, msg = AIRMED._try_build_model("adapted_model = 42")
        @test m == 42 && msg == "model built"
        m, _ = AIRMED._try_build_model("adapted_circuit = :legacy")
        @test m == :legacy
        m, msg = AIRMED._try_build_model("x = 1")
        @test isnothing(m) && occursin("no `adapted_model`", msg)
        m, msg = AIRMED._try_build_model("this is not julia (")
        @test isnothing(m) && startswith(msg, "build failed")
    end

    @testset "propose_model_adaptation with an LLM: winnowing keeps the smallest adequate subset" begin
        data_times, data_states = generate_true_data(T_SPAN, 100)
        reply = openai_reply(proposal_json(by_hook("r1_parallel", "Resistor"; name = "rp"),
                                           by_hook("r1_to_cap", "Resistor"; name = "rs")))
        result = with_mock_llm([reply]) do url, _
            propose_model_adaptation(problem; data_times, data_states,
                                     api = :openai, base_url = url,
                                     max_retries = 0, fit_iters = 100, build_model = true)
        end
        @test result.success
        # Only cap.v is measured, so one series resistor already explains the
        # drift and the second proposal is winnowed away.
        @test length(result.proposals) == 1
        @test result.conversation[1].n_parsed == 2
        @test occursin("kept after", result.conversation[1].note)
        @test isnothing(result.adapted_model)   # generated code is a patch, not standalone
    end

    @testset "propose_model_adaptation: invalid LLM proposals fall back to enumeration" begin
        data_times, data_states = generate_true_data(T_SPAN, 50)
        bad = """{"proposals": [
            {"name": "a", "component_type": "Resistor", "port_a": ["r9", "p"], "port_b": ["r9", "n"]},
            {"name": "b", "component_type": "Resistor", "port_a": ["r1", "p"], "port_b": ["cap", "n"]}]}"""
        result = with_mock_llm([openai_reply(bad)]) do url, _
            @test_logs (:warn, r"unknown subsystem") (:warn, r"does not match any ComponentHook") (:warn, r"were invalid") match_mode = :any propose_model_adaptation(
                problem; data_times, data_states, api = :openai, base_url = url,
                max_retries = 0, fit_params = false)
        end
        @test result.conversation[1].n_parsed == 2
        @test startswith(only(result.proposals).rationale, "Demo mode")
    end

    @testset "propose_model_adaptation: explain_on_failure asks for an explanation instead" begin
        data_times, data_states = generate_true_data(T_SPAN, 50)
        explanation = openai_reply(AIRMED._escalation_to_json("the load drew more current",
                                                              "inspect the wiring"))
        result = with_mock_llm([openai_reply("I cannot decide."), explanation]) do url, reqs
            r = propose_model_adaptation(problem; data_times, data_states, api = :openai,
                                         base_url = url, max_retries = 0, explain_on_failure = true)
            @test length(reqs) == 2                    # proposal call + explanation call
            r
        end
        @test !result.success
        @test isempty(result.proposals)
        expl, rec, ok = AIRMED._parse_escalation_json(result.escalation)
        @test ok && expl == "the load drew more current" && rec == "inspect the wiring"
    end

    @testset "propose_model_adaptation: a repeated proposal is not refitted" begin
        data_times, data_states = generate_true_data(T_SPAN, 100)
        same = openai_reply(proposal_json(by_hook("r1_parallel", "Resistor")))
        # A resistor across R1 can only lower the resistance, while the true
        # circuit has a higher one, so attempt 1 retries. Attempt 2 gets the same
        # structure back and must advance to the next untried combination, a
        # capacitor at r1_parallel, and fit that.
        result = with_mock_llm([same]) do url, _
            @test_logs (:info, r"optimising 1 parameter\(s\): c_r1_parallel") match_mode = :any propose_model_adaptation(
                problem; data_times, data_states, api = :openai,
                base_url = url, max_retries = 1, fit_iters = 50)
        end
        @test length(result.conversation) == 2
    end

    @testset "agent_supervise! with an LLM decision" begin
        cfg(url) = SupervisionConfig(use_agent = true, agent_api = :openai, agent_base_url = url,
                                     escalate_on_repeated_drift = false,
                                     ask_human_on_uncertainty = false)
        drift = DriftResult(true, 1, 0.1, 0.2, :cusum, Dict{Symbol, Any}())
        for (reply, expected) in (("continue", :continue), ("UPDATE.", :update), ("escalate", :escalate))
            with_mock_llm([openai_reply(reply)]) do url, _
                st = AgentState()
                @test agent_supervise!(st, [0.1], drift, cfg(url)) == expected
                @test occursin("LLM agent decision", last(st.log))
            end
        end
        with_mock_llm([openai_reply("banana")]) do url, _
            @test (@test_logs (:warn, r"unrecognised decision") agent_supervise!(
                AgentState(), [0.1], drift, cfg(url))) == :escalate
        end
        with_mock_llm(["oops"]; status = 500) do url, _
            @test (@test_logs (:warn, r"call failed") match_mode = :any agent_supervise!(
                AgentState(), [0.1], drift, cfg(url))) == :escalate
        end
    end

    @testset "supervisor and explanation backends without a key" begin
        st, drift = AgentState(), DriftResult(true, 1, 0.1, 0.2, :cusum, Dict{Symbol, Any}())
        base = (; use_agent = true, escalate_on_repeated_drift = false, ask_human_on_uncertainty = false)
        @test (@test_logs (:warn, r"no Anthropic API key") agent_supervise!(
            st, [0.1], drift, SupervisionConfig(; base..., agent_api = :anthropic))) == :update
        @test (@test_logs (:warn, r"No OpenAI API key") agent_supervise!(
            st, [0.1], drift, SupervisionConfig(; base..., agent_api = :openai))) == :escalate
        @test agent_supervise!(st, [0.1], drift, SupervisionConfig(; base..., agent_api = :unknown)) == :update

        @test startswith(explain_for_user("x", SupervisionConfig(agent_api = :anthropic)), "[anthropic stub")
        @test startswith(explain_for_user("x", SupervisionConfig(agent_api = :openai)), "[openai stub")
    end

    @testset "explain_for_user via an OpenAI-compatible endpoint" begin
        with_mock_llm([openai_reply("It drifted because of X.")]) do url, reqs
            cfg = SupervisionConfig(agent_api = :ollama, agent_base_url = url)
            @test explain_for_user("explain", cfg) == "It drifted because of X."
            @test only(reqs) == "/v1/chat/completions"
        end
        with_mock_llm(["down"]; status = 500) do url, _
            cfg = SupervisionConfig(agent_api = :openai, agent_base_url = url)
            @test startswith(explain_for_user("explain", cfg), "[openai error")
        end
    end
end
