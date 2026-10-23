# gemma: simple example, case 149

Counterfactual benchmark facts, not claims about the people or works named.

| Question | Expected entity | H0 | H1 | S0 | S1 | Clean C | No-op C+ |
|---|---|---|---|---|---|---|---|
| Which religion is Giovanni Papini affiliated with? | Q47280 | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion |
| Who founded Abrahamic religion? | Q9181 | Abraham | Abraham | Abraham | Abraham | Abraham | Abraham |
| Which religion is Albert of Saxony affiliated with? | Q47280 | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion | Abrahamic religion |
| Who founded the religion associated with Giovanni Papini? | Q9181 | Abraham | Jesus Christ | Abraham | Abraham | Abraham | Abraham |
| What is the name of the founder of the religion that Giovanni Papini believed in? | Q9181 | Abraham | Abraham | Abraham | Abraham | Abraham | Abraham |
| From whom was the religion that Giovanni Papini adhered to founded? | Q9181 | Abraham | Abraham | Abraham | Abraham | Abraham | Abraham |

## Full starting records and corrections

### C

```text
Initial record:
Giovanni Papini is affiliated with the religion of Abrahamic religion.
Albert of Saxony is affiliated with the religion of Abrahamic religion.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.
```

### C+

```text
Initial record:
Giovanni Papini is affiliated with the religion of Abrahamic religion.
Albert of Saxony is affiliated with the religion of Abrahamic religion.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.

Corrections:
SET Giovanni Papini is affiliated with the religion of Abrahamic religion.
SET Albert of Saxony is affiliated with the religion of Abrahamic religion.
```

### H0

```text
Initial record:
Giovanni Papini is affiliated with the religion of Catholic Church.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.
Albert of Saxony is affiliated with the religion of Abrahamic religion.

Corrections:
SET Giovanni Papini is affiliated with the religion of Abrahamic religion.
SET Albert of Saxony is affiliated with the religion of Abrahamic religion.
```

### H1

```text
Initial record:
Albert of Saxony is affiliated with the religion of Abrahamic religion.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.
Giovanni Papini is affiliated with the religion of Catholic Church.

Corrections:
SET Giovanni Papini is affiliated with the religion of Abrahamic religion.
SET Albert of Saxony is affiliated with the religion of Abrahamic religion.
```

### S0

```text
Initial record:
Albert of Saxony is affiliated with the religion of Catholic Church.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.
Giovanni Papini is affiliated with the religion of Abrahamic religion.

Corrections:
SET Giovanni Papini is affiliated with the religion of Abrahamic religion.
SET Albert of Saxony is affiliated with the religion of Abrahamic religion.
```

### S1

```text
Initial record:
Giovanni Papini is affiliated with the religion of Abrahamic religion.
Catholic Church was founded by Jesus Christ.
Abrahamic religion was founded by Abraham.
Albert of Saxony is affiliated with the religion of Catholic Church.

Corrections:
SET Giovanni Papini is affiliated with the religion of Abrahamic religion.
SET Albert of Saxony is affiliated with the religion of Abrahamic religion.
```
