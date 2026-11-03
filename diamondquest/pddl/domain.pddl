;; Diamond Quest -- STRIPS planning domain for a single explorer.
;; Problems are generated from the live game around the player (see diamondquest/pddl_planner.py).

(define (domain diamond-quest)
  (:requirements :strips)
  (:predicates
    (at ?c)            ; the explorer stands on cell ?c
    (adj ?a ?b)        ; ?a and ?b are orthogonal neighbours
    (open ?c)          ; floor, already dug, or fire
    (dirt ?c)          ; diggable dirt
    (cracked ?c)       ; weak wall, breakable with the hammer
    (gem ?c)           ; a diamond lies on ?c
    (has-gem))

  (:action move
    :parameters (?from ?to)
    :precondition (and (at ?from) (adj ?from ?to) (open ?to))
    :effect (and (at ?to) (not (at ?from))))

  (:action dig
    :parameters (?from ?to)
    :precondition (and (at ?from) (adj ?from ?to) (dirt ?to))
    :effect (and (at ?to) (not (at ?from))))

  (:action hammer-wall
    :parameters (?from ?to)
    :precondition (and (at ?from) (adj ?from ?to) (cracked ?to))
    :effect (and (open ?to) (not (cracked ?to))))

  (:action collect
    :parameters (?c)
    :precondition (and (at ?c) (gem ?c))
    :effect (and (has-gem) (not (gem ?c)))))
