import type { DemoEntry } from './demo';
import { cloudsEntry } from '../demos/clouds';
import { oceanEntry } from '../demos/ocean';
import { watercolourEntry } from '../demos/watercolour';
import { magicEntry } from '../demos/magic';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [oceanEntry, cloudsEntry, watercolourEntry, magicEntry];
