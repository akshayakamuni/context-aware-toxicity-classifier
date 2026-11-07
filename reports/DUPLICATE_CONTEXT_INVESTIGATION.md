# CAD v1.1 - Duplicate Context Investigation

## Overview
- Duplicate text groups: 78
- Individual duplicate records: 261
- Groups with different official labels: 11

## Parent Comment Availability
parent_status
same_official_split_parent       180
parent_missing_from_metadata      68
parent_not_in_official_splits     13

## Context Annotation Distribution
context_annotation
NA                 245
PreviousContent     12
CurrentContent       4

## Label Conflict Groups
### 'Based'
- Labels: Neutral | PersonDirectedAbuse
- Splits: test | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_missing_from_metadata | same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Based.'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_not_in_official_splits | same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Bullshit'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | test
- Different parent IDs: True
- Different threads: True
- Parent statuses: same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Exactly'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_missing_from_metadata | parent_not_in_official_splits | same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Get out'
- Labels: Neutral | PersonDirectedAbuse
- Splits: dev | test | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_missing_from_metadata | parent_not_in_official_splits | same_official_split_parent
- Context annotations: CurrentContent | NA

### 'No u'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Same'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Seconded.'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: dev | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: same_official_split_parent
- Context annotations: NA | PreviousContent

### 'This but unironically'
- Labels: IdentityDirectedAbuse | Neutral
- Splits: test | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Yes'
- Labels: AffiliationDirectedAbuse | IdentityDirectedAbuse | Neutral
- Splits: dev | test | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_missing_from_metadata | same_official_split_parent
- Context annotations: NA | PreviousContent

### 'Yes.'
- Labels: Neutral | PersonDirectedAbuse
- Splits: dev | test | train
- Different parent IDs: True
- Different threads: True
- Parent statuses: parent_missing_from_metadata | same_official_split_parent
- Context annotations: NA | PreviousContent

## Important Limitations

- Different labels do not automatically indicate annotation errors.
- Parent text availability does not guarantee complete conversation history.
- Metadata context annotations are not the official supervised labels.
- This investigation does not modify or remove any dataset records.
- Context relationships must be interpreted carefully before model use.