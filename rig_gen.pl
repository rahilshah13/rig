% rig_gen.pl
% Generative Procedural Rigging Grammar with Valid JSON AST Output
% Compatible with Trealla Prolog (zero-dependency, unmoduled).

:- set_prolog_flag(double_quotes, chars).

%% ---------------------------------------------------------
%% Public API
%% ---------------------------------------------------------

verify_rigging_program(String) :-
    phrase(program, String).

write_rigging_program(Index) :-
    (   number(Index), integer(Index) ->
        actions(Actions),
        length(Actions, Len),
        N is ((Index - 1) mod Len) + 1,
        nth1(N, Actions, ActionName),
        get_action_params(ActionName, Freq, Amp),
        bones(Bones),
        setup_call_cleanup(
            open('script.json', write, Stream),
            write_json_scene(Stream, Bones, ActionName, Freq, Amp),
            close(Stream)
        )
    ;   
        write_rigging_program_by_seconds(Index)
    ).

write_rigging_program(Frequency, Amplitude) :-
    bones(Bones),
    setup_call_cleanup(
        open('script.json', write, Stream),
        write_json_scene_custom(Stream, Bones, Frequency, Amplitude),
        close(Stream)
    ).

write_rigging_program_by_seconds(Seconds) :-
    bones(Bones),
    setup_call_cleanup(
        open('script.json', write, Stream),
        write_json_scene_seconds(Stream, Bones, Seconds),
        close(Stream)
    ).

%% ---------------------------------------------------------
%% Extended Random & Parameterized API for Batch Variations
%% ---------------------------------------------------------

write_random_rigging_program(Index) :-
    setrand(Index),
    actions(Actions),
    length(Actions, Len),
    random(1, Len + 1, RndIndex),
    nth1(RndIndex, Actions, ActionName),
    random(0.5, 4.5, Freq),
    random(0.1, 0.55, Amp),
    bones(Bones),
    format(atom(Filename), 'rig_~w.json', [Index]),
    setup_call_cleanup(
        open(Filename, write, Stream),
        write_json_scene(Stream, Bones, ActionName, Freq, Amp),
        close(Stream)
    ),
    setup_call_cleanup(
        open('script.json', write, Stream2),
        write_json_scene(Stream2, Bones, ActionName, Freq, Amp),
        close(Stream2)
    ).

write_rigging_program_with_seconds(Index, Seconds) :-
    bones(Bones),
    format(atom(Filename), 'rig_~w.json', [Index]),
    setup_call_cleanup(
        open(Filename, write, Stream),
        write_json_scene_seconds(Stream, Bones, Seconds),
        close(Stream)
    ).

actions([bowing, walking, waving, running, ducking, jumping, punching, kicking, dodging]).

get_action_params(bowing, 0.9, 0.3).
get_action_params(walking, 1.5, 0.35).
get_action_params(waving, 2.0, 0.4).
get_action_params(running, 2.8, 0.45).
get_action_params(ducking, 3.5, 0.3).
get_action_params(jumping, 4.2, 0.4).
get_action_params(punching, 3.6, 0.45).
get_action_params(kicking, 4.0, 0.4).
get_action_params(dodging, 2.5, 0.35).

%% ---------------------------------------------------------
%% Bone Hierarchy Definitions (FK Tree)
%% ---------------------------------------------------------
bone_parent('Spine', 'Hips').
bone_parent('Chest', 'Spine').
bone_parent('Neck', 'Chest').
bone_parent('Head', 'Neck').
bone_parent('LeftShoulder', 'Chest').
bone_parent('LeftUpperArm', 'LeftShoulder').
bone_parent('LeftLowerArm', 'LeftUpperArm').
bone_parent('LeftHand', 'LeftLowerArm').
bone_parent('RightShoulder', 'Chest').
bone_parent('RightUpperArm', 'RightShoulder').
bone_parent('RightLowerArm', 'RightUpperArm').
bone_parent('RightHand', 'RightLowerArm').
bone_parent('LeftUpperLeg', 'Hips').
bone_parent('LeftLowerLeg', 'LeftUpperLeg').
bone_parent('LeftFoot', 'LeftLowerLeg').
bone_parent('RightUpperLeg', 'Hips').
bone_parent('RightLowerLeg', 'RightUpperLeg').
bone_parent('RightFoot', 'RightLowerLeg').

bones([
    'Hips', 'Spine', 'Chest', 'Neck', 'Head',
    'LeftShoulder', 'LeftUpperArm', 'LeftLowerArm', 'LeftHand',
    'RightShoulder', 'RightUpperArm', 'RightLowerArm', 'RightHand',
    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
    'RightUpperLeg', 'RightLowerLeg', 'RightFoot'
]).

get_parent_json_val(Part, ParentStr) :-
    (   bone_parent(Part, Parent) 
    ->  format(atom(ParentStr), '"~w"', [Parent])
    ;   ParentStr = 'null'
    ).

%% ---------------------------------------------------------
%% Valid JSON Scene Serializer Loop with Strict Comma Separation
%% ---------------------------------------------------------

write_json_scene(Stream, Bones, Action, Freq, Amp) :-
    write(Stream, '[\n'),
    write_bones_with_commas(Stream, Bones, Action, Freq, Amp),
    write(Stream, '\n]\n').

write_json_scene_custom(Stream, Bones, Freq, Amp) :-
    write_json_scene(Stream, Bones, walking, Freq, Amp).

write_json_scene_seconds(Stream, Bones, Seconds) :-
    write(Stream, '[\n'),
    write_bones_seconds_with_commas(Stream, Bones, Seconds),
    write(Stream, '\n]\n').

write_bones_with_commas(_, [], _, _, _).
write_bones_with_commas(Stream, [Bone], Action, Freq, Amp) :-
    !,
    write_bone_json(Stream, Bone, Action, Freq, Amp).
write_bones_with_commas(Stream, [Bone | Rest], Action, Freq, Amp) :-
    write_bone_json(Stream, Bone, Action, Freq, Amp),
    write(Stream, ',\n'),
    write_bones_with_commas(Stream, Rest, Action, Freq, Amp).

write_bones_seconds_with_commas(_, [], _).
write_bones_seconds_with_commas(Stream, [Bone], Seconds) :-
    !,
    write_bone_seconds_json(Stream, Bone, Seconds).
write_bones_seconds_with_commas(Stream, [Bone | Rest], Seconds) :-
    write_bone_seconds_json(Stream, Bone, Seconds),
    write(Stream, ',\n'),
    write_bones_seconds_with_commas(Stream, Rest, Seconds).

write_bone_json(Stream, Bone, Action, Freq, Amp) :-
    get_parent_json_val(Bone, ParentVal),
    format(Stream, '  {\n    "bone": "~w",\n    "parent": ~w,\n    "transformations": [\n', [Bone, ParentVal]),
    emit_bone_transforms_json(Stream, Bone, Action, Freq, Amp),
    write(Stream, '\n    ]\n  }').

write_bone_seconds_json(Stream, Bone, Seconds) :-
    get_parent_json_val(Bone, ParentVal),
    format(Stream, '  {\n    "bone": "~w",\n    "parent": ~w,\n    "transformations": [\n', [Bone, ParentVal]),
    format(Stream, '      {"type": "translate", "values": [0.0, ~w, 0.0]},\n', [Seconds]),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": 0.5, "amp": ~w}\n', [Seconds]),
    write(Stream, '    ]\n  }').


%% ---------------------------------------------------------
%% Granular Transform AST Emitters for Actions
%% ---------------------------------------------------------

emit_bone_transforms_json(Stream, 'Spine', bowing, Freq, Amp) :- !,
    Bow is Amp * 0.8,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Bow]).
emit_bone_transforms_json(Stream, 'Chest', bowing, Freq, Amp) :- !,
    Bow is Amp * 0.6,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Bow]).
emit_bone_transforms_json(Stream, 'Head', bowing, Freq, Amp) :- !,
    Bow is Amp * 0.4,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Bow]).

emit_bone_transforms_json(Stream, 'Hips', walking, Freq, Amp) :- !,
    Double is Freq * 2, Bounce is Amp * 0.15, Sway is Amp * 0.2,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w},\n', [Double, Bounce]),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w}', [Freq, Sway]).
emit_bone_transforms_json(Stream, 'LeftUpperLeg', walking, Freq, Amp) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Amp]).
emit_bone_transforms_json(Stream, 'RightUpperLeg', walking, Freq, Amp) :- !,
    Opp is -Amp,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Opp]).
emit_bone_transforms_json(Stream, 'LeftUpperArm', walking, Freq, Amp) :- !,
    Arm is -(Amp * 0.6),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Arm]).
emit_bone_transforms_json(Stream, 'RightUpperArm', walking, Freq, Amp) :- !,
    ArmOpp is Amp * 0.6,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, ArmOpp]).

emit_bone_transforms_json(Stream, 'LeftUpperArm', waving, Freq, Amp) :- !,
    Wave is Amp * 1.5,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w}', [Freq, Wave]).
emit_bone_transforms_json(Stream, 'LeftLowerArm', waving, Freq, Amp) :- !,
    Elbow is Amp * 1.2,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w}', [Freq, Elbow]).
emit_bone_transforms_json(Stream, 'Chest', waving, Freq, Amp) :- !,
    Sway is Amp * 0.2,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w}', [Freq, Sway]).

emit_bone_transforms_json(Stream, 'Hips', running, Freq, Amp) :- !,
    Double is Freq * 2, Bounce is Amp * 0.4,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w}', [Double, Bounce]).
emit_bone_transforms_json(Stream, 'LeftUpperLeg', running, Freq, Amp) :- !,
    Run is Amp * 1.4,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Run]).
emit_bone_transforms_json(Stream, 'RightUpperLeg', running, Freq, Amp) :- !,
    RunOpp is -(Amp * 1.4),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, RunOpp]).
emit_bone_transforms_json(Stream, 'LeftLowerLeg', running, Freq, Amp) :- !,
    Knee is Amp * 1.3,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Knee]).
emit_bone_transforms_json(Stream, 'RightLowerLeg', running, Freq, Amp) :- !,
    KneeOpp is -(Amp * 1.3),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, KneeOpp]).
emit_bone_transforms_json(Stream, 'LeftUpperArm', running, Freq, Amp) :- !,
    Arm is -(Amp * 1.1),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Arm]).
emit_bone_transforms_json(Stream, 'RightUpperArm', running, Freq, Amp) :- !,
    ArmOpp is Amp * 1.1,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, ArmOpp]).

emit_bone_transforms_json(Stream, 'Hips', ducking, Freq, Amp) :- !,
    Bounce is Amp * 0.5,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w}', [Freq, Bounce]).
emit_bone_transforms_json(Stream, 'LeftUpperLeg', ducking, Freq, Amp) :- !,
    Leg is Amp * 0.9,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Leg]).
emit_bone_transforms_json(Stream, 'RightUpperLeg', ducking, Freq, Amp) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Amp]).
emit_bone_transforms_json(Stream, 'Spine', ducking, Freq, Amp) :- !,
    Spine is Amp * 0.4,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Spine]).

emit_bone_transforms_json(Stream, 'Hips', jumping, Freq, _) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": 0.8}', [Freq]).
emit_bone_transforms_json(Stream, 'LeftUpperLeg', jumping, Freq, _) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": 0.6}', [Freq]).
emit_bone_transforms_json(Stream, 'RightUpperLeg', jumping, Freq, _) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": 0.6}', [Freq]).
emit_bone_transforms_json(Stream, 'LeftUpperArm', jumping, Freq, _) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": -0.7}', [Freq]).
emit_bone_transforms_json(Stream, 'RightUpperArm', jumping, Freq, _) :- !,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": -0.7}', [Freq]).

emit_bone_transforms_json(Stream, 'Spine', punching, Freq, Amp) :- !,
    Twist is Amp * 0.9,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w}', [Freq, Twist]).
emit_bone_transforms_json(Stream, 'Chest', punching, Freq, Amp) :- !,
    ChestTwist is -(Amp * 1.1),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "y", "freq": ~w, "amp": ~w}', [Freq, ChestTwist]).
emit_bone_transforms_json(Stream, 'LeftUpperArm', punching, Freq, Amp) :- !,
    Jab is Amp * 1.8,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Jab]).
emit_bone_transforms_json(Stream, 'RightUpperArm', punching, Freq, Amp) :- !,
    Cross is -(Amp * 1.8),
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Cross]).

emit_bone_transforms_json(Stream, 'Hips', kicking, Freq, Amp) :- !,
    Stab is Amp * 0.3,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w}', [Freq, Stab]).
emit_bone_transforms_json(Stream, 'LeftUpperLeg', kicking, Freq, Amp) :- !,
    Kick is Amp * 2.2,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Kick]).
emit_bone_transforms_json(Stream, 'LeftLowerLeg', kicking, Freq, Amp) :- !,
    Snap is Amp * 1.8,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Snap]).

emit_bone_transforms_json(Stream, 'Hips', dodging, Freq, Amp) :- !,
    Sustain is Amp * 0.8, Slip is Amp * 0.7,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w},\n', [Freq, Sustain]),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": ~w}', [Freq, Slip]).
emit_bone_transforms_json(Stream, 'Spine', dodging, Freq, Amp) :- !,
    Weave is Amp * 0.9,
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "z", "freq": ~w, "amp": ~w}', [Freq, Weave]).

% Default fallback
emit_bone_transforms_json(Stream, _, _, Freq, _) :-
    format(Stream, '      {"type": "translate", "values": [0.0, 0.0, 0.0]},\n', []),
    format(Stream, '      {"type": "oscillate", "axis": "x", "freq": ~w, "amp": 0.03}', [Freq]).


%% ---------------------------------------------------------
%% Legacy Program Validator (for verify_rigging_program)
%% ---------------------------------------------------------
program --> ws, lines, ws.
program --> ws.
lines --> line, ws, lines.
lines --> line.
line --> identifier(_), optional_parent, ws, ":", ws1, command.
optional_parent --> ws, "[parent:", ws, identifier(_), ws, "]".
optional_parent --> [].
command --> identifier(_), "(", ws, number_list, ws, ")", opt_semi.
number_list --> number, ws, ",", ws, number_list.
number_list --> number.
number_list --> [].
opt_semi --> ws, ";".
opt_semi --> [].
ws --> ws_char, ws.
ws --> [].
ws1 --> ws_char, ws.
ws_char --> [C], { is_space(C) }.
is_space(' '). is_space('\t'). is_space('\n'). is_space('\r').
identifier(Atom) --> [C], { is_alpha(C) }, identifier_rest(Chars), { atom_chars(Atom, [C | Chars]) }.
identifier_rest([C | Rest]) --> [C], { is_alnum(C) ; C = '_' }, identifier_rest(Rest).
identifier_rest([]) --> [].
number --> opt_sign, digits, opt_fraction.
opt_sign --> "-" ; "+" ; [].
digits --> [C], { is_digit(C) }, digits_rest.
digits_rest --> [C], { is_digit(C) }, digits_rest.
digits_rest --> [].
opt_fraction --> ".", digits.
opt_fraction --> [].
is_alpha(C) :- C @>= 'a', C @=< 'z' ; C @>= 'A', C @=< 'Z'.
is_digit(C) :- C @>= '0', C @=< '9'.
is_alnum(C) :- is_alpha(C) ; is_digit(C).