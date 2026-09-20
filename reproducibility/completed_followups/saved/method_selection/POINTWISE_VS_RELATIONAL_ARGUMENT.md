# Pointwise evaluation and relational source-history sufficiency

## Formal observation

Let h denote an update history, S_q(h) its declared current facts relevant to a
downstream query q, and F(h,q) the response produced by the resulting system at
a fixed evaluation interface. A finite battery E(h) evaluates that post-update
system separately, for example by checking factual correctness, preservation,
and selected consequences. It may establish E(h1)=pass and E(h2)=pass when
S_q(h1)=S_q(h2). Source-history sufficiency instead asks whether
F(h1,q)=F(h2,q) for the specified history and query families whenever
S_q(h1)=S_q(h2). Equal declared facts are the operational matching condition;
they do not assert equality of unobserved internal states. For stochastic
systems the corresponding property concerns response distributions, rather
than equality of independently sampled strings.

Agreement on the battery's finite pointwise observables does not, by itself,
force agreement on an untested downstream computation: two systems or reached
states may agree on every measured observable while differing elsewhere. This
is a limitation of what those measurements certify, not a deep impossibility
theorem. If the battery already requires the same uniquely correct answer to q
under both histories, it does imply agreement for that q. Exhaustive testing of
a finite target family, or stronger structural assumptions connecting tested
observables to all target responses, can also imply the property without a
separately named matched-history test. The non-implication applies where neither
the relevant relational condition nor a sufficient stronger guarantee is
established. Even equal aggregate accuracy is weaker than equal pointwise
outcomes; it is not a substitute for matched-history invariance.

## Plain-English Introduction version

Checking an update one history at a time and checking whether its past has been
screened off are different questions. A model can satisfy the checks we ran
without those checks determining its answer to every other downstream query.
Matched-history evaluation asks directly whether different superseded records
still matter after the declared current facts are held fixed. The empirical
importance of that extra information for choosing a method must be demonstrated,
not inferred from this logical observation.

## Why can't standard evaluation already see this?

It can if it tests the relevant histories and downstream answers or supplies a
stronger guarantee; an ordinary finite pointwise battery that does neither does
not certify the cross-history invariance in question.

## Empirical connection

The accompanying METHOD_SELECTION_REANALYSIS.md tests whether this distinction
changes a method choice after measured ordinary performance is practically
equivalent. The structural observation alone establishes neither an empirical
full-battery counterexample nor a decision consequence, and must not be used
to rescue an unsupported method-selection comparison.
