import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def person_has_parents(people, person):
    if people.get(person)["father"] is not None and people.get(person)["mother"] is not None:
        return True
    else:
        return False


def calculate_parent_prob(parent, one_gene, two_genes):
    if parent in one_gene:
        return 0.5
    elif parent in two_genes:
        return 1 - PROBS["mutation"]
    else:
        return PROBS["mutation"]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    probability = 0.0

    for person in people:
        if person in one_gene:
            number_of_genes = 1
        elif person in two_genes:
            number_of_genes = 2
        else:
            number_of_genes = 0

        if not person_has_parents(people, person):
            probability = join_probs(probability, PROBS["gene"][number_of_genes])
        else:
            father_prob, mother_prob = get_prob_on_parents(one_gene, person, two_genes, people)
            if number_of_genes == 0:
                probability = join_probs(probability, (1 - father_prob) * (1 - mother_prob))
            elif number_of_genes == 1:
                probability = join_probs(probability, father_prob * (1 - mother_prob) + mother_prob * (1 - father_prob))
            elif number_of_genes == 2:
                probability = join_probs(probability, father_prob * mother_prob)

        has_trait = person in have_trait
        probability = join_probs(probability, PROBS["trait"][number_of_genes][has_trait])

    return probability


def join_probs(probability, prob):
    if probability == 0:
        return prob
    else:
        return probability * prob


def get_prob_on_parents(one_gene, person, two_genes, people):
    father = people.get(person)["father"]
    mother = people.get(person)["mother"]
    father_prob = calculate_parent_prob(father, one_gene, two_genes)
    mother_prob = calculate_parent_prob(mother, one_gene, two_genes)

    return father_prob, mother_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][1] += p
        else:
            probabilities[person]["trait"][0] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities:
        probabilities[person]["gene"] = normalize_dict(probabilities[person]["gene"])
        probabilities[person]["trait"] = normalize_dict(probabilities[person]["trait"])


def normalize_dict(probs_dict):
    sum_probabilities = 0.0

    for key in probs_dict:
        sum_probabilities += probs_dict.get(key)

    alfa = 1 / sum_probabilities

    for key in probs_dict:
        new_value = alfa * probs_dict.get(key)
        probs_dict[key] = new_value

    return probs_dict


if __name__ == "__main__":
    main()
