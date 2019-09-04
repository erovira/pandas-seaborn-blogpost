# %%

import numpy as np
import pandas as pd
import seaborn as sns
import timeit

# %% [markdown]

"""
## Pandas
"""

# %%

# Load dataset
titanic = sns.load_dataset("titanic")

# %%

titanic

# %%

pd.Series([1, 90, "hey", np.nan], index=["a", "B", "C", "d"])

# %%

pd.DataFrame({"day": [17, 30], "month": [1, 12], "year": [2010, 2017]})

# %%

titanic.info()

# %%

# Returns a scalar
# titanic.ix[4, 'age']
titanic.at[4, "age"]

# %%

# Returns a Series of name 'age', and the age values associated
# to the index labels 4 and 5
# titanic.ix[[4, 5], 'age']
titanic.loc[[4, 5], "age"]

# %%

# Returns a Series of name '4', and the age and fare values
# associated to that row.
# titanic.ix[4, ['age', 'fare']]
titanic.loc[4, ["age", "fare"]]

# %%

titanic.loc[[4, 5], ["age", "fare"]]

# %%

A = pd.Series(["a", "b", "c", "d", "e", "f"])
A_filter = pd.Series([False, True, False, True, False, True, True])
A[A_filter]

# %%

titanic[
    (titanic.sex == "female")
    & (titanic["class"].isin(["First", "Third"]))
    & (titanic.age > 30)
    & (titanic.survived == 0)
]

# %%

s1 = pd.Series([10, 20, 30, np.nan], index=["a", "b", "c", "d"])
s2 = pd.Series([100, 200, 300, 400], index=["e", "d", "c", "b"])
s1.add(s2)

# %%

s1.align(s2)

# %%

df1 = pd.DataFrame({"a": [1]})
df2 = pd.DataFrame({"b": [2]})

# 1) Incorrect: Raises AttributeError since concat is not in the DataFrame namespace
df1.concat(df2)


# %%

# 2) Correct
pd.concat([df1, df2], sort=True)

# %%

# 3) Correct, and equivalent to 2)
df1.append(df2, sort=True)

# %%

# 4) Incorrect: append does not accept the 'axis' arg.
#    Should use pd.concat with axis='columns'
df2.append(df2, axis="columns")


# %%

# Generate a small DataFrame of cities with their population,
# and fake ages (years with the city status)
towns_dic = {
    "name": ["Southampton", "Cherbourg", "Queenstown", "Montevideo"],
    "country": ["United Kingdom", "France", "United Kingdom", "Uruguay"],
    "population": [236900, 37121, 12347, 1305000],
    "age": [np.random.randint(500, 1000) for _ in range(4)],
}
towns_df = pd.DataFrame(towns_dic)
towns_df

# %%

(
    titanic.merge(
        towns_df,
        left_on="embark_town",
        right_on="name",
        how="left",
        indicator=True,
        suffixes=("_passenger", "_city"),
    )
).head()
# 'head' takes the last n elements of the DataFrame

# %%

bins = [0, 12, 17, 60, np.inf]
labels = ["child", "teenager", "adult", "elder"]
age_groups = pd.cut(titanic.age, bins, labels=labels)
titanic["age_group"] = age_groups
titanic

# %%

groups = titanic.groupby(["age_group", "alone"])
type(groups)


# %%

groups.size()

# %%

100 * groups.size() / len(titanic)

# %%


def is_old_func(row):
    return row["age"] > 60


titanic["is_old"] = titanic.apply(is_old_func, axis="columns")
titanic

# %%


def is_old_func_series(value):
    return value > 60


titanic["is_old"] = titanic["age"].apply(is_old_func_series)
titanic

# %%

titanic.eval("is_old = age > 60", inplace=True)
titanic

# %%

%timeit -n100 titanic['is_old'] = titanic.apply(is_old_func, axis='columns') # noqa
%timeit -n100 titanic['is_old'] = titanic['age'].apply(is_old_func_series) # noqa
%timeit -n100 titanic.eval('is_old = age > 60', inplace=True) # noqa

# %%

p_titanic = titanic.drop_duplicates("age").pivot(
    index="age", columns="class", values="fare"
)
p_titanic.tail(3)

# %%

pt_titanic = titanic.pivot_table(index="age", columns="class", values="fare")
pt_titanic.tail(3)


# %%

titanic.pivot_table(
    index="embark_town", columns="age_group", values="fare", aggfunc=np.median
)

# %%

pd.melt(
    p_titanic.reset_index(),
    id_vars="age",
    var_name="class_renamed",
    value_vars=["First", "Second", "Third"],
    value_name="fare",
).tail(3)


# %% [markdown]

"""
Note : At the time of the blogpost, the columns class and deck were of
type object instead of category as they are now.

This causes all sort of issues, including
[this](https://github.com/pandas-dev/pandas/issues/19136) one.
We can work around it with a simple type cast.
"""

# %%

p_titanic_copy = p_titanic.copy()
p_titanic_copy.columns = p_titanic_copy.columns.astype(str)
pd.melt(
    p_titanic_copy.reset_index(),
    id_vars="age",
    var_name="class_renamed",
    value_vars=["First", "Second", "Third"],
    value_name="fare",
).tail(3)


# %%

titanic.pivot_table(index="embark_town", columns="age_group", aggfunc=np.median).fare


# %%

groups.size().unstack()


# %%

groups.size().unstack(level=0)

# %% [markdown]
"""
## Seaborn
"""

# %%

sns.distplot(titanic["age"].dropna())

# %%

g = sns.FacetGrid(titanic, row="survived", col="class")
g.map(sns.distplot, "age")

# %%

sns.jointplot(data=titanic, x="age", y="fare", kind="reg", color="g")

# %%

df = titanic.pivot_table(
    index="embark_town", columns="age_group", values="fare", aggfunc=np.median
)
sns.heatmap(df, annot=True, fmt=".1f")


# %%

sns.heatmap(titanic.corr(), annot=True, fmt=".2f")
